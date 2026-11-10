"""System One request compilation, candidate prompt rendering, and typed response formatting for Jev."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from lmml.core.llm.jev.metrics import choice_confidence, score_confidence
from lmml.core.llm.kev.api import Choice, Noul, Score, SystemOneRequest, SystemOneResponse


def _render(value: Any) -> str:
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )


def _description(value: Any, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, (str, dict, list)):
        raise TypeError("instructions and descriptions must be text, an object, or an array")
    json.dumps(value, allow_nan=False)
    return _render(value)


def _normalize_definition(definition: Any) -> dict[str, Any]:
    if isinstance(definition, (Noul, Choice, Score)):
        return definition.model_dump(exclude_none=False)
    if isinstance(definition, Mapping):
        return dict(definition)
    raise ValueError("question definitions must be mappings or Noul/Choice/Score instances")


def compile_request(state: Any, questions: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compile a shared state and typed questions into isolated model records.

    Records contain no target. `answer_keys` and `legend` are software-only
    metadata; `candidate_prompts` never passes them or question IDs to a model.
    """
    if not isinstance(state, (str, dict, list)):
        raise TypeError("state must be text, a JSON object, or an array")
    state_copy = json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False))
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("questions must be a nonempty mapping")
    records: list[dict[str, Any]] = []
    for question_id, raw_def in questions.items():
        if not isinstance(question_id, str):
            raise TypeError("question IDs must be strings")
        definition = _normalize_definition(raw_def)
        kind = definition.get("type")
        if kind not in ("choice", "score", "noul"):
            raise ValueError("question type must be choice, score, or noul")
        question = _description(definition.get("instructions"))
        assert question is not None
        criteria = definition.get("criteria")
        record: dict[str, Any] = {
            "id": question_id,
            "state": state_copy,
            "kind": kind,
            "question": question,
        }
        if kind == "choice":
            if not isinstance(criteria, Mapping) or not 1 <= len(criteria) <= 255:
                raise ValueError("Choice requires between 1 and 255 candidates")
            if any(not isinstance(key, str) for key in criteria):
                raise ValueError("Choice candidate names must be strings")
            descriptions = [_description(value, optional=True) for value in criteria.values()]
            record["answer_keys"] = list(criteria)
            record["options"] = [
                name if description is None else f"{name}: {description}"
                for name, description in zip(criteria, descriptions)
            ]
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("Score requires an array of 2 to 10 descriptive levels")
            record["options"] = [_description(level) for level in criteria]
            record["answer_keys"] = [str(index) for index in range(len(criteria))]
            record["legend"] = dict(zip(record["answer_keys"], json.loads(json.dumps(criteria))))
        else:
            if criteria is not None:
                if not isinstance(criteria, Mapping) or not set(criteria).issubset({"true", "false"}):
                    raise ValueError("Noul criteria must contain only true and false descriptions")
                for key, label in (("true", "Yes"), ("false", "No")):
                    if key in criteria and criteria[key] is not None:
                        description = _description(criteria[key])
                        record["question"] += f"\n{label} means: {description}"
            record["options"] = ["no", "yes"]
            record["answer_keys"] = ["false", "true"]
        records.append(record)
    return records


def candidate_prefix(record: Mapping[str, Any]) -> str:
    """Render the shared state + question prefix once for all candidates."""
    return f"Context:\n{_render(record['state'])}\n\nQuestion: {_render(record['question'])}\n"


def candidate_prompts_from_prefix(record: Mapping[str, Any], prefix: str) -> list[str]:
    """Append candidate-specific suffixes to a shared rendered prefix."""
    if record["kind"] == "noul":
        return [prefix + "Is the answer to this question yes? Answer Yes or No."]
    return [
        prefix + f"Proposed answer: {_render(option)}\nIs this proposed answer correct? Answer Yes or No."
        for option in record["options"]
    ]


def candidate_prompts(record: Mapping[str, Any]) -> list[str]:
    """Render independent candidates using only declared model input fields."""
    return candidate_prompts_from_prefix(record, candidate_prefix(record))


def apply_candidate_chat_template(tokenizer: Any, prompt: str) -> str:
    """Wrap a candidate prompt with the Qwen3.8 chat template (enable_thinking=False)."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return str(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        except TypeError:
            return str(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_response(
    records: Sequence[Mapping[str, Any]],
    probabilities: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Return typed answers; invalid model probabilities fail validation."""
    if len(records) != len(probabilities) or not records:
        raise ValueError("records and probability rows must have equal nonzero length")
    answers: dict[str, Any] = {}
    for record, values in zip(records, probabilities):
        question_id, kind = str(record["id"]), str(record["kind"])
        if question_id in answers:
            raise ValueError("duplicate question ID in response records")
        keys = list(record.get("answer_keys") or record.get("options") or [])
        if kind == "noul" and not keys:
            keys = ["false", "true"]
        if len(values) != len(keys):
            raise ValueError("probability count must match the declared answer space")
        probs = [float(value) for value in values]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probs):
            raise ValueError("probabilities must be finite and in [0, 1]")
        total = sum(probs)
        if not math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError("probabilities must sum to one")
        probs = [value / total for value in probs]
        if kind == "noul":
            if [str(k).lower() for k in keys] not in (["false", "true"], ["no", "yes"]):
                raise ValueError("Noul probabilities must be ordered false, true")
            answer: dict[str, Any] = {"type": kind, "noul": probs[1]}
        elif kind == "choice":
            selected = max(range(len(probs)), key=probs.__getitem__)
            answer = {
                "type": kind,
                "choice": keys[selected],
                "probabilities": dict(zip(keys, probs)),
                "confidence": choice_confidence(probs),
            }
        elif kind == "score":
            legend = record.get("legend") or dict(zip(keys, record.get("options", keys)))
            answer = {
                "type": kind,
                "score": sum(index * value for index, value in enumerate(probs)),
                "probabilities": dict(zip(keys, probs)),
                "confidence": score_confidence(probs),
                "legend": legend,
            }
        else:
            raise ValueError(f"unknown record kind: {kind!r}")
        answers[question_id] = answer
    return {"answers": answers}


__all__ = [
    "Choice",
    "Noul",
    "Score",
    "SystemOneRequest",
    "SystemOneResponse",
    "apply_candidate_chat_template",
    "candidate_prefix",
    "candidate_prompts",
    "candidate_prompts_from_prefix",
    "compile_request",
    "format_response",
]
