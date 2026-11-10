"""Kev decision model: a causal Qwen backbone plus an option pointer readout."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
OPT_NONE, OPT_DECIDE = -1, -2
MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048
SERVE_MAX_STATE, SERVE_MAX_BRANCH = 65536, 65536 + 8192
ROW_PASS_TOKENS = 16384
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
QWEN35_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
]


class ContextOverflow(ValueError):
    def __init__(self, message: str, state_tokens: int | None = None, max_state: int | None = None) -> None:
        super().__init__(message)
        self.state_tokens = state_tokens
        self.max_state = max_state


def training_context(max_state: int = MAX_STATE) -> dict[str, int]:
    if not MAX_STATE <= max_state <= SERVE_MAX_STATE:
        raise ValueError(f"max_state must be in [{MAX_STATE}, {SERVE_MAX_STATE}]")
    extra = max_state - MAX_STATE
    return {"max_state": max_state, "max_branch": MAX_BRANCH + extra, "max_packed": MAX_PACKED + extra}


def load_tokenizer(name: str, revision: str | None = None) -> Any:
    return AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True)


def pad_id(tokenizer: Any) -> int:
    return int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)


def text_config(config: Any) -> Any:
    """Qwen3.5 wraps its language config inside a multimodal top-level config."""
    nested = getattr(config, "text_config", None)
    return nested if nested is not None and hasattr(nested, "hidden_size") else config


def is_hybrid(config: Any) -> bool:
    config = text_config(config)
    return "linear_attention" in set(getattr(config, "layer_types", None) or [])


def user_tokens(tokenizer: Any, text: str) -> list[int]:
    """Tokenize untrusted text with delimiter-looking strings neutralized."""
    safe = _SPECIAL_RE.sub(r"<¦\1¦>", str(text))
    return list(tokenizer(safe, add_special_tokens=False).input_ids)


def encode(
    tokenizer: Any,
    record: dict[str, Any],
    *,
    max_state: int = MAX_STATE,
    max_branch: int = MAX_BRANCH,
    strict: bool = True,
    option_isolation: bool = False,
) -> dict[str, Any]:
    """Encode one materialized record into state and isolated question branches."""
    state_tokens = user_tokens(tokenizer, record["state"])
    state_length = len(state_tokens) + 1
    if strict and state_length > max_state:
        raise ContextOverflow(
            f"state exceeds {max_state} tokens: {state_length}",
            state_tokens=state_length,
            max_state=max_state,
        )
    state_ids = [tokenizer.convert_tokens_to_ids(SPECIAL[0]), *state_tokens[: max_state - 1]]
    ids = list(state_ids)
    seg = [0] * len(state_ids)
    pos = list(range(len(state_ids)))
    opt = [OPT_NONE] * len(state_ids)
    q_id, o_id, c_id, d_id = (tokenizer.convert_tokens_to_ids(token) for token in SPECIAL[1:])
    decide_indices: list[int] = []
    option_indices: list[list[int]] = []
    for branch_index, question in enumerate(record["questions"], start=1):
        instruction = [q_id, *user_tokens(tokenizer, question.get("instr", ""))]
        spans = [[o_id, *user_tokens(tokenizer, option), c_id] for option in question["options"]]
        branch = [*instruction, *(token for span in spans for token in span), d_id]
        if strict and len(branch) > max_branch - len(state_ids):
            raise ContextOverflow(
                f"branch too long: {len(branch)} tokens with a {len(state_ids)}-token state "
                f"(row limit {max_branch})"
            )
        base = len(ids)
        state_prefix = len(state_ids)
        branch_positions = list(range(state_prefix, state_prefix + len(branch)))
        branch_options = [OPT_NONE] * len(instruction)
        for option_index, span in enumerate(spans):
            branch_options.extend([option_index] * len(span))
        branch_options.append(OPT_DECIDE)
        option_ends: list[int] = []
        cursor = len(instruction)
        for span in spans:
            cursor += len(span)
            option_ends.append(cursor - 1)
        ids.extend(branch)
        seg.extend([branch_index] * len(branch))
        pos.extend(branch_positions)
        opt.extend(branch_options)
        decide_indices.append(base + len(branch) - 1)
        option_indices.append([base + end for end in option_ends])
    return {
        "ids": ids,
        "seg": seg,
        "pos": pos,
        "opt": opt,
        "option_isolation": bool(option_isolation),
        "decide_idx": decide_indices,
        "opt_idx": option_indices,
        "labels": [int(question["label"]) for question in record["questions"]],
        "targets": [question.get("target") for question in record["questions"]],
        "qtypes": [question.get("qtype", "choice") for question in record["questions"]],
        "keys": [question.get("keys") for question in record["questions"]],
        "state_tokens": state_length,
        "state_truncated": state_length > max_state,
    }


def branch_mask(seg: Sequence[int], device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return branch_mask_batch([seg], device, dtype)


def branch_mask_batch(
    segs: Sequence[Sequence[int]],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    *,
    opts: Sequence[Sequence[int]] | None = None,
    length: int | None = None,
) -> torch.Tensor:
    size = max(max(len(segment) for segment in segs), length or 0)
    segment = torch.full((len(segs), size), -1, dtype=torch.long, device=device)
    for row, values in enumerate(segs):
        segment[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
    causal = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    same = (segment[:, None, :] == segment[:, :, None]) | (segment[:, None, :] == 0)
    valid_key = (segment != -1)[:, None, :]
    allow = causal[None] & same & valid_key
    if opts is not None:
        option = torch.full((len(opts), size), OPT_NONE, dtype=torch.long, device=device)
        for row, values in enumerate(opts):
            option[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        key_is_option = option[:, None, :] >= 0
        query_is_decide = option[:, :, None] == OPT_DECIDE
        same_option = option[:, None, :] == option[:, :, None]
        allow = allow & (~key_is_option | query_is_decide | same_option)
    allow = allow | torch.eye(size, dtype=torch.bool, device=device)[None]
    return torch.zeros(len(segs), size, size, dtype=dtype, device=device).masked_fill(
        ~allow, torch.finfo(dtype).min
    )[:, None]


def rows_of(encoded: dict[str, Any]) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    state_length = encoded["seg"].count(0)
    rows: list[dict[str, Any]] = []
    start = state_length
    for branch_index, (decide, option_indices) in enumerate(
        zip(encoded["decide_idx"], encoded["opt_idx"]), start=1
    ):
        end = decide + 1
        if encoded["seg"][start] != branch_index or encoded["seg"][end - 1] != branch_index:
            raise ValueError("branch layout mismatch")
        rows.append(
            {
                "ids": encoded["ids"][start:end],
                "pos": encoded["pos"][start:end],
                "decide": decide - start,
                "opts": [index - start for index in option_indices],
                "opt": encoded["opt"][start:end],
            }
        )
        start = end
    return encoded["ids"][:state_length], encoded["pos"][:state_length], rows


class PointerHead(nn.Module):
    def __init__(self, hidden_size: int, pointer_dim: int = 256) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_size, pointer_dim)
        self.k = nn.Linear(hidden_size, pointer_dim)
        self.scale = 1.0 / math.sqrt(pointer_dim)
        self.temperature = 1.0

    def forward(self, decide: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        logits = (self.k(options) @ self.q(decide)) * self.scale
        return logits if self.training or self.temperature == 1.0 else logits / self.temperature


def _resolve_lora_targets(model: nn.Module) -> list[str]:
    leaves = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    targets = [name for name in QWEN35_LORA_TARGETS if name in leaves]
    if not targets:
        raise ValueError("no Qwen LoRA target modules found")
    return targets


class KevDecisionModel(nn.Module):
    """A prefill-only causal backbone with one pointer distribution per question."""

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Any,
        *,
        mode: str = "lora",
        torch_dtype: torch.dtype | str = torch.bfloat16,
        attn_implementation: str = "sdpa",
        gradient_checkpointing: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] | None = None,
        pointer_dim: int = 256,
        option_isolation: bool = False,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        mode = mode.lower()
        if mode not in {"lora", "full"}:
            raise ValueError("mode must be 'lora' or 'full'")
        self.mode = mode
        self.option_isolation = bool(option_isolation)
        self.pad_id = pad_id(tokenizer)
        self.torch_dtype = torch_dtype
        if isinstance(torch_dtype, str):
            self.torch_dtype = {
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
                "fp32": torch.float32,
                "float32": torch.float32,
                "auto": "auto",
            }.get(torch_dtype.lower(), torch.bfloat16)
        try:
            loaded = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=self.torch_dtype,
                attn_implementation=attn_implementation,
                device_map=None,
                trust_remote_code=trust_remote_code,
            )
            self.lm = getattr(loaded, "model", loaded)
        except (OSError, ValueError, KeyError):
            loaded = AutoModel.from_pretrained(
                model_name_or_path,
                dtype=self.torch_dtype,
                attn_implementation=attn_implementation,
                device_map=None,
                trust_remote_code=trust_remote_code,
            )
            self.lm = getattr(loaded, "language_model", loaded)
        self.hybrid = is_hybrid(self.lm.config)
        if self.hybrid:
            from lmml.core.llm.jev.model import ensure_qwen35_cpu_linear_attention_fallback

            ensure_qwen35_cpu_linear_attention_fallback()
        if self.hybrid and self.option_isolation:
            raise ValueError("option_isolation is unavailable on hybrid Qwen3.5 backbones")
        if gradient_checkpointing:
            self.lm.config.use_cache = False
            self.lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if mode == "lora":
            from peft import LoraConfig, TaskType, get_peft_model

            targets = list(lora_target_modules or _resolve_lora_targets(self.lm))
            leaves = {name.rsplit(".", 1)[-1] for name, _ in self.lm.named_modules()}
            targets = [name for name in targets if name in leaves]
            if not targets:
                raise ValueError("none of the configured LoRA target modules exist in the backbone")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=targets,
            )
            self.lm = get_peft_model(self.lm, peft_config)
            if hasattr(self.lm, "enable_input_require_grads"):
                self.lm.enable_input_require_grads()
        else:
            self.lm.requires_grad_(True)
        self.head = PointerHead(int(text_config(self.lm.config).hidden_size), pointer_dim)
        self.train()

    def encode(self, tokenizer: Any, record: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return encode(tokenizer, record, option_isolation=self.option_isolation, **kwargs)

    def _pad_rows(self, rows: Sequence[tuple[Sequence[int], Sequence[int]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = max(len(ids) for ids, _ in rows)
        device = next(self.parameters()).device
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long, device=device)
        positions = torch.zeros((len(rows), width), dtype=torch.long, device=device)
        attention = torch.zeros((len(rows), width), dtype=torch.long, device=device)
        for index, (row_ids, row_positions) in enumerate(rows):
            ids[index, : len(row_ids)] = torch.tensor(row_ids, dtype=torch.long, device=device)
            positions[index, : len(row_positions)] = torch.tensor(row_positions, dtype=torch.long, device=device)
            attention[index, : len(row_ids)] = 1
        return ids, positions, attention

    def forward_batch(self, encodings: Sequence[dict[str, Any]]) -> list[list[torch.Tensor]]:
        """Return one list of option logits per record."""
        if not encodings:
            return []
        rows: list[tuple[list[int], list[int]]] = []
        readouts: list[tuple[int, int, list[int]]] = []
        segments: list[list[int]] = []
        option_ids: list[list[int]] = []
        for record_index, encoded in enumerate(encodings):
            state_ids, state_positions, branches = rows_of(encoded)
            for branch in branches:
                rows.append(([*state_ids, *branch["ids"]], [*state_positions, *branch["pos"]]))
                segments.append([0] * len(state_ids) + [1] * len(branch["ids"]))
                option_ids.append([OPT_NONE] * len(state_ids) + list(branch["opt"]))
                readouts.append(
                    (
                        record_index,
                        len(state_ids) + branch["decide"],
                        [len(state_ids) + index for index in branch["opts"]],
                    )
                )
        ids, positions, attention = self._pad_rows(rows)
        if self.option_isolation:
            attention = branch_mask_batch(
                segments,
                ids.device,
                dtype=next(self.parameters()).dtype,
                opts=option_ids,
                length=ids.shape[1],
            )
        outputs = self.lm(input_ids=ids, attention_mask=attention, position_ids=positions)
        hidden = outputs.last_hidden_state.float()
        logits: list[list[torch.Tensor]] = [[] for _ in encodings]
        device = hidden.device
        for row_index, (record_index, decide_index, option_indices) in enumerate(readouts):
            logits[record_index].append(
                self.head(hidden[row_index, decide_index], hidden[row_index, torch.tensor(option_indices, device=device)])
            )
        return logits

    @torch.no_grad()
    def probabilities(self, encodings: Sequence[dict[str, Any]]) -> list[list[torch.Tensor]]:
        was_training = self.training
        self.eval()
        try:
            logits = self.forward_batch(encodings)
            return [[F.softmax(values, dim=-1).cpu() for values in record] for record in logits]
        finally:
            self.train(was_training)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


__all__ = [
    "MAX_BRANCH",
    "MAX_PACKED",
    "MAX_STATE",
    "OPT_DECIDE",
    "OPT_NONE",
    "QWEN35_LORA_TARGETS",
    "ROW_PASS_TOKENS",
    "SERVE_MAX_BRANCH",
    "SERVE_MAX_STATE",
    "SPECIAL",
    "ContextOverflow",
    "KevDecisionModel",
    "PointerHead",
    "branch_mask",
    "branch_mask_batch",
    "encode",
    "is_hybrid",
    "load_tokenizer",
    "pad_id",
    "rows_of",
    "text_config",
    "training_context",
    "user_tokens",
]
