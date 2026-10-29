"""Multi-turn Agentic RL with tool observations masked from policy gradients."""

from __future__ import annotations

import json
import re
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from lmml.core.llm.rl_common.reward_adapter import RewardAdapter, RewardConfig
from lmml.core.llm.rl_common.rollout import TorchRolloutEngine
from lmml.core.llm.rl_common.tools import execute_tool, parse_tool_calls
from lmml.models.llm.builders import build_causal_lm


def _final_answer(text: str) -> str:
    pieces = re.split(r"</tool_call>|</think>", text)
    return pieces[-1].strip() if pieces else text.strip()


def _gt_score(answer: str, gt: list[str]) -> float:
    if not gt:
        return 0.0
    hits = 0
    for item in gt:
        if str(item).lower() in answer.lower():
            hits += 1
    return 2.5 * hits / len(gt)


class CausalLMAgentRLLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Any,
        learning_rate: float = 3e-7,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        model_config: dict[str, Any] | None = None,
        max_turns: int = 3,
        max_new_tokens: int = 128,
        max_prompt_length: int = 1024,
        num_generations: int = 4,
        beta: float = 0.1,
        epsilon_high: float = 5.0,
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
        self.max_turns = int(max_turns)
        self.max_new_tokens = int(max_new_tokens)
        self.max_prompt_length = int(max_prompt_length)
        self.num_generations = int(num_generations)
        self.beta = float(beta)
        self.epsilon_high = float(epsilon_high)

    def _rollout_one(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        gt: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], bool, float]:
        conversations = [dict(m) for m in messages]
        response_ids: list[int] = []
        response_mask: list[int] = []
        old_logps: list[float] = []
        turn_texts: list[str] = []
        unfinished = False
        prompt_len = 0
        for turn in range(self.max_turns):
            prompt = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools or None,
            )
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_ids)
            input_ids = torch.tensor([prompt_ids + response_ids], device=self.device)
            rollout = self.rollout_engine.rollout(
                input_ids,
                torch.ones_like(input_ids),
                num_generations=1,
                max_new_tokens=self.max_new_tokens,
            )
            valid_len = int(rollout.completion_mask[0].sum().item())
            new_ids = rollout.completion_ids[0, :valid_len].tolist()
            new_logps = rollout.per_token_logps[0, :valid_len].tolist()
            new_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            turn_texts.append(new_text)
            response_ids.extend(new_ids)
            response_mask.extend([1] * len(new_ids))
            old_logps.extend(new_logps)
            calls = parse_tool_calls(new_text)
            if not calls:
                break
            unfinished = turn == self.max_turns - 1
            conversations.append({"role": "assistant", "content": new_text})
            for call in calls:
                name = str(call.get("name", ""))
                args = call.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                result = execute_tool(name, args if isinstance(args, dict) else {})
                conversations.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result or {"error": "tool not found"}, ensure_ascii=False),
                    }
                )
            next_prompt = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=not unfinished,
                tools=tools or None,
            )
            next_ids = self.tokenizer.encode(next_prompt, add_special_tokens=False)
            observation = next_ids[prompt_len :]
            # Observation tokens are provided by the environment; mask them from PPO.
            response_ids.extend(observation)
            response_mask.extend([0] * len(observation))
            old_logps.extend([0.0] * len(observation))
        answer = _final_answer(turn_texts[-1] if turn_texts else "")
        initial_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools or None,
        )
        parsed_turns = [parse_tool_calls(text) for text in turn_texts]
        all_calls = [call for calls in parsed_turns for call in calls]
        malformed = sum(
            abs(text.count("<tool_call>") - text.count("</tool_call>"))
            for text in turn_texts
        )
        reward = _gt_score(answer, gt) - 0.5 * malformed
        reward -= 0.5 if unfinished else 0.0
        if all_calls:
            valid_calls = sum(
                1
                for call in all_calls
                if execute_tool(
                    str(call.get("name", "")),
                    call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                )
                is not None
            )
            reward += 0.5 * valid_calls / max(1, len(all_calls))
        else:
            reward += self.reward_scorer.score(initial_prompt, answer)
        reward = max(-3.0, min(3.0, reward))
        initial_ids = self.tokenizer.encode(initial_prompt, add_special_tokens=False)
        abs_ids = torch.tensor([initial_ids + response_ids], device=self.device)
        mask = torch.tensor(
            [[0] * len(initial_ids) + response_mask], device=self.device
        )
        old = torch.tensor(
            [[0.0] * len(initial_ids) + old_logps], device=self.device, dtype=torch.float
        )
        return abs_ids, mask, old, [prompt_len], unfinished, reward

    def _ensure_reference(self) -> None:
        if self.reference is None:
            import copy

            self.reference = copy.deepcopy(self.model).eval().requires_grad_(False)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        self._ensure_reference()
        raw_messages = batch.get("messages", batch.get("messages_json"))
        messages = [
            json.loads(item) if isinstance(item, str) else item for item in raw_messages
        ]
        tools = [json.loads(item) if isinstance(item, str) else item for item in batch.get("tools_json", batch.get("tools", []))]
        gt = batch["gt"]
        samples = []
        for message, tool_def, ground_truth in zip(messages, tools, gt):
            for _ in range(self.num_generations):
                samples.append(self._rollout_one(message, tool_def or [], ground_truth))
        width = max(sample[0].size(1) for sample in samples)
        pad = int(self.tokenizer.pad_token_id)
        ids = torch.full((len(samples), width), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros((len(samples), width), dtype=torch.long, device=self.device)
        old = torch.zeros((len(samples), width), dtype=torch.float, device=self.device)
        rewards = torch.tensor([sample[5] for sample in samples], device=self.device)
        for i, (ids_i, mask_i, old_i, _, _, _) in enumerate(samples):
            ids[i, : ids_i.size(1)] = ids_i
            mask[i, : mask_i.size(1)] = mask_i
            old[i, : old_i.size(1)] = old_i
        grouped = rewards.view(-1, self.num_generations)
        mean = grouped.mean(-1).repeat_interleave(self.num_generations)
        std = grouped.std(-1, unbiased=False).repeat_interleave(self.num_generations)
        advantages = (rewards - mean) / (std + 1e-4)
        outputs = self.model(input_ids=ids)
        aux_loss = getattr(outputs, "aux_loss", ids.new_zeros(()))
        logits = F.log_softmax(outputs.logits[:, :-1].float(), dim=-1)
        targets = ids[:, 1:]
        logps = logits.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            ref_logps = F.log_softmax(self.reference(input_ids=ids).logits[:, :-1].float(), dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
        old_shift = torch.cat(
            (torch.zeros((len(samples), 1), device=self.device), old[:, :-1]), dim=1
        )[:, : logps.size(1)]
        ratio = torch.exp((logps - old_shift).clamp(-20, 20))
        clamped = torch.clamp(ratio, max=self.epsilon_high).detach()
        token_mask = torch.cat(
            (torch.zeros((len(samples), 1), device=self.device), mask[:, :-1]), dim=1
        )[:, : logps.size(1)].float()
        per_token = -(clamped * advantages[:, None] * logps - self.beta * (torch.exp(ref_logps - logps) - (ref_logps - logps) - 1))
        loss = (per_token * token_mask).sum() / token_mask.sum().clamp(min=1) + aux_loss
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/reward", rewards.mean(), prog_bar=True)
        self.log("train/policy_loss", loss, prog_bar=True)
        self.log("train/avg_response_len", mask.sum(-1).float().mean(), prog_bar=True)
        return loss

    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> dict[str, Any]:
        from lmml.core.llm.rl_common.predict import predict_from_batch

        return predict_from_batch(self.model, self.tokenizer, batch, self.max_new_tokens)

    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.model.parameters(), lr=getattr(self, "learning_rate", 3e-7))


__all__ = ["CausalLMAgentRLLightningModule"]
