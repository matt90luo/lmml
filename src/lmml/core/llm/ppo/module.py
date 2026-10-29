"""PPO with an actor/critic on MiniMind and token-level GAE."""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from lmml.core.llm.rl_common.reward_adapter import RewardAdapter, RewardConfig
from lmml.core.llm.rl_common.rollout import TorchRolloutEngine
from lmml.models.llm.builders import build_causal_lm
from lmml.models.llm.minimind import MiniMindCriticForCausalLM


class CausalLMPPOLightningModule(pl.LightningModule):
    automatic_optimization = False

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Any,
        learning_rate: float = 3e-7,
        critic_learning_rate: float = 5e-7,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        model_config: dict[str, Any] | None = None,
        max_new_tokens: int = 128,
        max_prompt_length: int = 768,
        gamma: float = 1.0,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        value_clip: float = 0.2,
        vf_coef: float = 0.5,
        kl_coef: float = 0.02,
        update_iters: int = 2,
        mini_batch_size: int = 1,
        reward_mode: str = "rule_only",
        reward_model_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.tokenizer = tokenizer
        self.actor = build_causal_lm(
            model_name_or_path=model_name_or_path,
            mode="full",
            torch_dtype=torch_dtype,
            use_gradient_checkpointing=False,
            trust_remote_code=trust_remote_code,
            model_config=model_config,
        )
        import copy

        self.reference = None
        self.critic = MiniMindCriticForCausalLM(copy.deepcopy(self.actor.config))
        self.critic.load_state_dict(self.actor.state_dict(), strict=False)
        self.reward_scorer = RewardAdapter(
            RewardConfig(mode=reward_mode, model_path=reward_model_path)
        )
        self.max_update_iters = int(update_iters)
        self.max_mini_batch_size = int(mini_batch_size)
        self.rollout_engine = TorchRolloutEngine(self.actor, tokenizer)
        self.max_new_tokens = int(max_new_tokens)
        self.max_prompt_length = int(max_prompt_length)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.clip_epsilon = float(clip_epsilon)
        self.value_clip = float(value_clip)
        self.vf_coef = float(vf_coef)
        self.kl_coef = float(kl_coef)
        self.update_iters = int(update_iters)
        self.mini_batch_size = int(mini_batch_size)

    def _token_logps(
        self,
        model: torch.nn.Module,
        ids: torch.Tensor,
        mask: torch.Tensor,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        outputs = model(input_ids=ids, attention_mask=mask, logits_to_keep=ids.size(1) - start + 1)
        logits = outputs.logits[:, : ids.size(1) - start, :].float()
        targets = ids[:, start:]
        values = F.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
        return values, getattr(outputs, "aux_loss", None)

    def _ensure_reference(self) -> None:
        if self.reference is None:
            import copy

            self.reference = copy.deepcopy(self.actor).eval().requires_grad_(False)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        self._ensure_reference()
        self.actor.train()
        self.critic.train()
        prompts = list(batch["prompt"])
        prompt_ids = [self.tokenizer.encode(text, add_special_tokens=False)[-self.max_prompt_length :] for text in prompts]
        pad_id = int(self.tokenizer.pad_token_id)
        max_len = max(len(ids) for ids in prompt_ids)
        ids_tensor = torch.full((len(prompt_ids), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids_tensor)
        for i, ids in enumerate(prompt_ids):
            ids_tensor[i, max_len - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            mask[i, max_len - len(ids) :] = 1
        device = self.device
        ids_tensor, mask = ids_tensor.to(device), mask.to(device)
        rollout = self.rollout_engine.rollout(
            ids_tensor,
            mask,
            num_generations=1,
            max_new_tokens=self.max_new_tokens,
        )
        gen_ids = rollout.output_ids
        prompt_lens = rollout.prompt_lens
        completion_mask = rollout.completion_mask
        prompt_len = int(prompt_lens[0].item())
        old_logps = rollout.per_token_logps.detach()
        rewards = torch.tensor(
            [self.reward_scorer.score(p, c) for p, c in zip(prompts, rollout.completions)],
            device=device,
        )
        full_mask = torch.cat((mask, completion_mask), 1)
        with torch.no_grad():
            values = self.critic(gen_ids, attention_mask=full_mask)
            resp_values = values[:, prompt_len : prompt_len + completion_mask.size(1)]
        with torch.no_grad():
            ref_logps, _ = self._token_logps(self.reference, gen_ids, full_mask, prompt_len)

        token_rewards = torch.zeros_like(old_logps)
        lengths = completion_mask.sum(-1).long()
        valid = lengths > 0
        rows = torch.arange(gen_ids.size(0), device=device)
        token_rewards[rows[valid], lengths[valid] - 1] = rewards[valid]
        advantages = torch.zeros_like(old_logps)
        last = torch.zeros(gen_ids.size(0), device=device)
        for t in reversed(range(old_logps.size(1))):
            next_value = resp_values[:, t + 1] if t + 1 < resp_values.size(1) else 0.0
            delta = token_rewards[:, t] + self.gamma * next_value - resp_values[:, t]
            last = delta + self.gamma * self.lam * last
            advantages[:, t] = last
        mask_float = completion_mask.float()
        mean = (advantages * mask_float).sum() / mask_float.sum().clamp(min=1)
        var = (((advantages - mean) ** 2) * mask_float).sum() / mask_float.sum().clamp(min=1)
        advantages = (advantages - mean) / torch.sqrt(var + 1e-8) * mask_float
        returns = advantages.detach() + resp_values
        old_values = resp_values

        actor_optimizer, critic_optimizer = self.optimizers()
        n = gen_ids.size(0)
        mb = max(1, min(self.mini_batch_size, n))
        for _ in range(self.update_iters):
            order = torch.randperm(n, device=device)
            for start in range(0, n, mb):
                inds = order[start : start + mb]
                # Use no KV-cache during PPO updates so repeated minibatch passes
                # do not retain the rollout graph or reuse stale caches.
                mb_logps, actor_aux = self._token_logps(self.actor, gen_ids[inds], full_mask[inds], prompt_len)
                ratio = torch.exp((mb_logps - old_logps[inds]).clamp(-20, 20))
                clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = (
                    torch.minimum(
                        ratio * advantages[inds],
                        clipped * advantages[inds],
                    )
                    * mask_float[inds]
                ).sum() / mask_float[inds].sum().clamp(min=1)
                log_ratio = mb_logps - ref_logps[inds]
                kl = (torch.exp(log_ratio) - log_ratio - 1) * mask_float[inds]
                policy_loss = policy_loss + self.kl_coef * kl.sum() / mask_float[inds].sum().clamp(min=1)
                if actor_aux is not None:
                    policy_loss = policy_loss + actor_aux
                mb_values = self.critic(gen_ids[inds], attention_mask=full_mask[inds])[:, prompt_len : prompt_len + old_logps.size(1)]
                value_loss = 0.5 * (
                    torch.maximum(
                        (mb_values - returns[inds]) ** 2,
                        (torch.clamp(mb_values, old_values[inds] - self.value_clip, old_values[inds] + self.value_clip) - returns[inds]) ** 2,
                    ) * mask_float[inds]
                ).sum() / mask_float[inds].sum().clamp(min=1)
                loss = policy_loss + self.vf_coef * value_loss
                actor_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)
                self.manual_backward(loss)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                actor_optimizer.step()
                critic_optimizer.step()
        self.log("train/loss", policy_loss.detach(), prog_bar=True)
        self.log("train/reward", rewards.mean(), prog_bar=True)
        self.log("train/policy_loss", policy_loss.detach())
        self.log("train/kl_ref", kl.detach().mean())
        self.log("train/avg_response_len", mask_float.sum(-1).mean())
        return policy_loss.detach()

    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> dict[str, Any]:
        from lmml.core.llm.rl_common.predict import predict_from_batch

        return predict_from_batch(self.actor, self.tokenizer, batch, self.max_new_tokens)

    def configure_optimizers(self) -> Any:
        actor = torch.optim.AdamW(self.actor.parameters(), lr=getattr(self, "learning_rate", 3e-7))
        critic = torch.optim.AdamW(self.critic.parameters(), lr=getattr(self, "critic_learning_rate", 5e-7))
        return actor, critic


__all__ = ["CausalLMPPOLightningModule"]
