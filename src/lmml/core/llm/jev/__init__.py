"""Qwen3.8-27B Jev Decision Model (independent candidate LoRA + FP32 linear head)."""

from lmml.core.llm.jev.api import (
    Choice,
    Noul,
    Score,
    SystemOneRequest,
    SystemOneResponse,
    apply_candidate_chat_template,
    candidate_prefix,
    candidate_prompts,
    candidate_prompts_from_prefix,
    compile_request,
    format_response,
)
from lmml.core.llm.jev.data import (
    JevArrowDataset,
    JevCollator,
    JevDataModule,
    JevPredictionCollator,
)
from lmml.core.llm.jev.metrics import (
    choice_confidence,
    evaluate_probabilities,
    evaluate_split_records,
    fit_temperature,
    score_confidence,
    sigmoid,
    softmax,
)
from lmml.core.llm.jev.model import DEFAULT_JEV_LORA_TARGETS, JevDecisionModel
from lmml.core.llm.jev.module import (
    JevLoraLightningModule,
    QwenLoraLightningModule,
    jev_nll_brier_loss,
)
from lmml.core.llm.jev.recipe import JevDecisionRecipe, JevStage3EvaluationCallback
from lmml.core.llm.jev.sdk import JevClient, TypeSafeClient

__all__ = [
    "DEFAULT_JEV_LORA_TARGETS",
    "Choice",
    "JevArrowDataset",
    "JevClient",
    "JevCollator",
    "JevDataModule",
    "JevDecisionModel",
    "JevDecisionRecipe",
    "JevLoraLightningModule",
    "JevPredictionCollator",
    "JevStage3EvaluationCallback",
    "Noul",
    "QwenLoraLightningModule",
    "Score",
    "SystemOneRequest",
    "SystemOneResponse",
    "TypeSafeClient",
    "apply_candidate_chat_template",
    "candidate_prefix",
    "candidate_prompts",
    "candidate_prompts_from_prefix",
    "choice_confidence",
    "compile_request",
    "evaluate_probabilities",
    "evaluate_split_records",
    "fit_temperature",
    "format_response",
    "jev_nll_brier_loss",
    "score_confidence",
    "sigmoid",
    "softmax",
]
