"""Direct Preference Optimization LightningModule."""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from lmml.models.llm.builders import build_causal_lm


class CausalLMDPOLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Any | None = None,
        learning_rate: float = 4e-8,
        beta: float = 0.15,
        torch_dtype: str = "bfloat16",
        use_gradient_checkpointing: bool = True,
        trust_remote_code: bool = True,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.tokenizer = tokenizer
        self.beta = float(beta)
        self.save_hyperparameters(ignore=["tokenizer"])
        self.policy = build_causal_lm(
            model_name_or_path=model_name_or_path,
            mode="full",
            torch_dtype=torch_dtype,
            use_gradient_checkpointing=use_gradient_checkpointing,
            trust_remote_code=trust_remote_code,
            model_config=model_config,
        )
        # The frozen reference shares the initial weights and is kept in eval mode.
        import copy

        self.reference = copy.deepcopy(self.policy)
        self.reference.eval().requires_grad_(False)

    def _logps(
        self,
        model: torch.nn.Module,
        ids: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        shift_labels = labels[:, 1:].contiguous()
        logits = outputs.logits[:, :-1, :].float()
        token_logps = F.log_softmax(logits, dim=-1).gather(
            2, shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)
        valid = (shift_labels != -100).float()
        aux_loss = getattr(outputs, "aux_loss", ids.new_zeros((), dtype=torch.float32))
        return (token_logps * valid).sum(dim=1), aux_loss

    def _loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        chosen_ids = batch["chosen_input_ids"]
        rejected_ids = batch["rejected_input_ids"]
        chosen_mask = batch["chosen_attention_mask"]
        rejected_mask = batch["rejected_attention_mask"]
        chosen_labels = batch["chosen_labels"]
        rejected_labels = batch["rejected_labels"]

        with torch.no_grad():
            ref_chosen, _ = self._logps(self.reference, chosen_ids, chosen_mask, chosen_labels)
            ref_rejected, _ = self._logps(self.reference, rejected_ids, rejected_mask, rejected_labels)
        policy_chosen, chosen_aux = self._logps(self.policy, chosen_ids, chosen_mask, chosen_labels)
        policy_rejected, rejected_aux = self._logps(self.policy, rejected_ids, rejected_mask, rejected_labels)
        margin = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
        loss = -F.logsigmoid(self.beta * margin).mean() + chosen_aux + rejected_aux
        return loss, margin

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, margin = self._loss(batch)
        self.log("train/loss", loss, prog_bar=True, batch_size=batch["chosen_input_ids"].size(0))
        self.log("train/dpo_loss", loss, prog_bar=True)
        self.log("train/reward_margin", margin.mean(), prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, margin = self._loss(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/reward_margin", margin.mean())
        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, margin = self._loss(batch)
        self.log("test/loss", loss, prog_bar=True, sync_dist=True)
        self.log("test/reward_margin", margin.mean(), sync_dist=True)
        return loss

    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> dict[str, Any]:
        from lmml.core.llm.rl_common.predict import predict_from_batch

        return predict_from_batch(self.policy, self.tokenizer, batch, 128)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.learning_rate)
        return optimizer


__all__ = ["CausalLMDPOLightningModule"]
