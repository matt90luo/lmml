"""Group Relative Policy Optimization and CISPO."""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from lmml.core.llm.rl_common.reward_adapter import RewardAdapter, RewardConfig
from lmml.core.llm.rl_common.rollout import TorchRolloutEngine
from lmml.models.llm.builders import build_causal_lm


class CausalLMGRPOLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Any,
        learning_rate: float = 3e-7,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        model_config: dict[str, Any] | None = None,
        max_prompt_length: int = 768,
        max_new_tokens: int = 128,
        num_generations: int = 6,
        beta: float = 0.1,
        epsilon: float = 0.2,
        epsilon_high: float = 5.0,
        loss_type: str = "cispo",
        reward_mode: str = "rule_only",
        reward_model_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.tokenizer = tokenizer
        self.model = build_causal_lm(
            model_name_or_path=model_name_or_path,
            mode="full",
            torch_dtype=torch_dtype,
            use_gradient_checkpointing=False,
            trust_remote_code=trust_remote_code,
            model_config=model_config,
        )
        self.reference = None
        self.reward_scorer = RewardAdapter(
            RewardConfig(mode=reward_mode, model_path=reward_model_path)
        )
        self.rollout_engine = TorchRolloutEngine(self.model, tokenizer)
        self.max_prompt_length = int(max_prompt_length)
        self.max_new_tokens = int(max_new_tokens)
        self.num_generations = int(num_generations)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.epsilon_high = float(epsilon_high)
        self.loss_type = str(loss_type).lower()

    def _logps(self, model: torch.nn.Module, ids: torch.Tensor, mask: torch.Tensor, start: int) -> torch.Tensor:
        outputs = model(input_ids=ids, attention_mask=mask)
        targets = ids[:, start:]
        return F.log_softmax(outputs.logits[:, start - 1 : -1, :], dim=-1).gather(
            2, targets.unsqueeze(-1)
        ).squeeze(-1)

    def _ensure_reference(self) -> None:
        if self.reference is None:
            import copy

            self.reference = copy.deepcopy(self.model).eval().requires_grad_(False)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        self._ensure_reference()
        prompts = list(batch["prompt"])
        ids = [self.tokenizer.encode(text, add_special_tokens=False)[-self.max_prompt_length :] for text in prompts]
        pad = int(self.tokenizer.pad_token_id)
        width = max(len(row) for row in ids)
        prompt_ids = torch.full((len(ids), width), pad, dtype=torch.long)
        prompt_mask = torch.zeros_like(prompt_ids)
        for i, row in enumerate(ids):
            prompt_ids[i, width - len(row) :] = torch.tensor(row)
            prompt_mask[i, width - len(row) :] = 1
        prompt_ids, prompt_mask = prompt_ids.to(self.device), prompt_mask.to(self.device)
        rollout = self.rollout_engine.rollout(
            prompt_ids,
            prompt_mask,
            num_generations=self.num_generations,
            max_new_tokens=self.max_new_tokens,
        )
        full_mask = torch.cat((prompt_mask.repeat(self.num_generations, 1), rollout.completion_mask), 1)
        start = int(rollout.prompt_lens[0].item())
        rewards = torch.tensor(
            [
                self.reward_scorer.score(prompts[index // self.num_generations], completion)
                for index, completion in enumerate(rollout.completions)
            ],
            device=self.device,
        ).view(len(prompts), self.num_generations)
        mean = rewards.mean(-1).repeat_interleave(self.num_generations)
        std = rewards.std(-1, unbiased=False).repeat_interleave(self.num_generations)
        advantages = (rewards.reshape(-1) - mean) / (std + 1e-4)
        old_logps = rollout.per_token_logps.detach()
        outputs = self.model(input_ids=rollout.output_ids, attention_mask=full_mask)
        targets = rollout.output_ids[:, start:]
        logps = F.log_softmax(outputs.logits[:, start - 1 : -1, :], dim=-1).gather(
            2, targets.unsqueeze(-1)
        ).squeeze(-1)
        aux_loss = getattr(outputs, "aux_loss", rollout.output_ids.new_zeros(()))
        with torch.no_grad():
            ref_outputs = self.reference(input_ids=rollout.output_ids, attention_mask=full_mask)
            ref_logps = F.log_softmax(ref_outputs.logits[:, start - 1 : -1, :], dim=-1).gather(
                2, targets.unsqueeze(-1)
            ).squeeze(-1)
        ratio = torch.exp((logps - old_logps).clamp(-20, 20))
        if self.loss_type == "cispo":
            clamped = torch.clamp(ratio, max=self.epsilon_high).detach()
            per_token = -(clamped * advantages[:, None] * logps - self.beta * (torch.exp(ref_logps - logps) - (ref_logps - logps) - 1))
        else:
            clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            per_token = -(torch.minimum(ratio * advantages[:, None], clipped * advantages[:, None]) - self.beta * (torch.exp(ref_logps - logps) - (ref_logps - logps) - 1))
        mask = rollout.completion_mask.float()
        loss = (per_token * mask).sum() / mask.sum().clamp(min=1) + aux_loss
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/reward", rewards.mean(), prog_bar=True)
        self.log("train/advantages_std", advantages.std(), prog_bar=True)
        self.log("train/policy_loss", loss, prog_bar=True)
        self.log("train/avg_response_len", mask.sum(-1).mean(), prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Validation keeps rollout cost bounded and reports the exact training loss.
        return self.training_step(batch, batch_idx)

    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> dict[str, Any]:
        from lmml.core.llm.rl_common.predict import predict_from_batch

        return predict_from_batch(self.model, self.tokenizer, batch, self.max_new_tokens)

    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.model.parameters(), lr=getattr(self, "learning_rate", 3e-7))


__all__ = ["CausalLMGRPOLightningModule"]
