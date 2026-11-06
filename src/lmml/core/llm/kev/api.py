"""Type-safe System One request primitives and answer serialization for Kev.

The decision model maps every question to one pointer-classification problem:
``Noul`` has two options, ``Choice`` has named options and ``Score`` has ordered
levels.  This module is intentionally dependency-light: it is shared by data
preparation, training, the in-process SDK, serving and prediction.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

JSONContent = str | dict | list | int | float | bool | None
MAX_OPTIONS = 255
EPSILON = 1e-9


class Noul(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: JSONContent = None
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check(self) -> Choice:
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    type: Literal["score"] = "score"
    instructions: JSONContent = None
    criteria: list[JSONContent] = Field(min_length=1, max_length=MAX_OPTIONS)


Question = Noul | Choice | Score


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = "kev-latest"
    questions: dict[str, Question] = Field(min_length=1)


class SystemOneResponse(BaseModel):
    model: str = "kev-latest"
    answers: dict[str, Any]
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = 0.0
    truncated: bool | None = None


def render(value: JSONContent, indent: int = 0) -> str:
    """Flatten strings, objects and arrays into stable labelled text."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render(item, indent + 1).lstrip()}" for item in value)
    return "\n".join(
        f"{pad}{key}:\n{render(item, indent + 1)}"
        if isinstance(item, (dict, list))
        else f"{pad}{key}: {render(item)}"
        for key, item in value.items()
    )


def option_text(name: str, description: JSONContent) -> str:
    return name if description is None or description == "" else f"{name}: {render(description)}"


MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE = re.compile(rf"\b(?:{MONTHS}) \d{{1,2}}, \d{{4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b")


def date_facts(text: str) -> str:
    """Return deterministic pairwise day differences for absolute dates in text."""
    found: list[tuple[str, datetime]] = []
    for match in _DATE.finditer(text):
        raw = match.group(0)
        try:
            value = datetime.strptime(raw, "%B %d, %Y") if "," in raw else datetime.strptime(raw, "%Y-%m-%d")
        except ValueError:
            continue
        if raw not in {name for name, _ in found}:
            found.append((raw, value))
    facts: list[str] = []
    for index, (left_text, left) in enumerate(found):
        for right_text, right in found[index + 1 :]:
            days = (right - left).days
            if days == 0:
                facts.append(f"{right_text} is the same day as {left_text}.")
            else:
                direction = "after" if days > 0 else "before"
                facts.append(
                    f"{right_text} is {abs(days)} day{'s' if abs(days) != 1 else ''} {direction} {left_text}."
                )
    return " ".join(facts)


def with_date_facts(state: JSONContent) -> JSONContent:
    facts = date_facts(render(state))
    if not facts:
        return state
    if isinstance(state, dict):
        return {**state, "date_facts": facts}
    if isinstance(state, list):
        return [*state, {"date_facts": facts}]
    return f"{state}\n\ndate_facts: {facts}"


def question_keys(qtype: str, criteria: Any) -> list[str]:
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(index) for index in range(len(criteria))]


def to_record(request: SystemOneRequest, *, date_facts_enabled: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert an API request into the internal ``state + questions`` record.

    Returns ``(record, metadata)`` where metadata is in the original question
    order and is used to map probability vectors back to API answers.
    """
    state = with_date_facts(request.state) if date_facts_enabled else request.state
    questions: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for question_id, question in request.questions.items():
        meta = {
            "id": question_id,
            "type": question.type,
            "keys": question_keys(question.type, question.criteria),
        }
        if question.type == "noul":
            criteria = question.criteria or {}
            options = [
                option_text("no", criteria.get("false")),
                option_text("yes", criteria.get("true")),
            ]
        elif question.type == "choice":
            options = [option_text(key, value) for key, value in question.criteria.items()]
        else:
            options = [render(value) for value in question.criteria]
            meta["legend"] = dict(zip(meta["keys"], options))
        questions.append(
            {
                "instr": render(question.instructions),
                "options": options,
                "qtype": question.type,
                "qid": question_id,
                "keys": meta["keys"],
            }
        )
        metadata.append(meta)
    return {"state": render(state), "questions": questions}, metadata


def materialize_labelled(record: dict[str, Any]) -> dict[str, Any]:
    """Validate a labelled JSONL record and attach integer labels/soft targets."""
    request = SystemOneRequest.model_validate(
        {"state": record["state"], "model": record.get("model", "kev-latest"), "questions": record["questions"]}
    )
    internal, metadata = to_record(request)
    for question, meta in zip(internal["questions"], metadata):
        raw = request.questions[meta["id"]]
        label = getattr(raw, "label", None)
        if isinstance(raw, dict):
            label = raw.get("label")
        # Pydantic models preserve unknown fields only when configured; labelled
        # suite records are dicts, so read them from the original payload.
        source_question = record["questions"][meta["id"]]
        if "label" not in source_question:
            raise ValueError(f"question {meta['id']!r} has no label")
        label = source_question["label"]
        if meta["type"] == "choice":
            question["label"] = meta["keys"].index(label)
        elif meta["type"] == "noul":
            question["label"] = int(bool(label))
        else:
            question["label"] = int(label)
            if not 0 <= question["label"] < len(meta["keys"]):
                raise ValueError(f"score label out of range for question {meta['id']!r}")
        target = source_question.get("target")
        if target is not None:
            if meta["type"] != "choice" or not isinstance(target, dict):
                raise ValueError("soft targets are supported for Choice questions only")
            values = [float(target.get(key, 0.0)) for key in meta["keys"]]
            total = sum(values)
            if total <= 0:
                raise ValueError(f"soft target for {meta['id']!r} must have positive mass")
            question["target"] = [value / total for value in values]
        question["src"] = source_question.get("src", "custom")
    return internal


def _normalize(probabilities: list[float]) -> list[float]:
    total = sum(probabilities)
    return [1.0 / len(probabilities)] * len(probabilities) if total == 0 else [value / total for value in probabilities]


def choice_confidence(probabilities: list[float]) -> float:
    count = len(probabilities)
    return 1.0 if count == 1 else (max(_normalize(probabilities)) - 1 / count) / (1 - 1 / count)


def score_confidence(probabilities: list[float]) -> float:
    count = len(probabilities)
    if count == 1:
        return 1.0
    normalized = _normalize(probabilities)
    mode = max(range(count), key=normalized.__getitem__)
    deviation = sum(abs(index - (count - 1) / 2) for index in range(count)) / count
    return max(0.0, 1.0 - sum(p * abs(index - mode) for index, p in enumerate(normalized)) / deviation)


def round_probability(value: float) -> float:
    return round(float(value), 4)


def to_answers(probabilities: list[list[float]], metadata: list[dict[str, Any]]) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    for probs, meta in zip(probabilities, metadata):
        if meta["type"] == "noul":
            answers[meta["id"]] = {"type": "noul", "noul": round_probability(probs[1])}
        elif meta["type"] == "choice":
            index = max(range(len(probs)), key=probs.__getitem__)
            answers[meta["id"]] = {
                "type": "choice",
                "choice": meta["keys"][index],
                "confidence": round_probability(choice_confidence(probs)),
                "probabilities": {key: round_probability(value) for key, value in zip(meta["keys"], probs)},
            }
        else:
            score = sum(index * value for index, value in enumerate(probs))
            answers[meta["id"]] = {
                "type": "score",
                "score": round_probability(score),
                "legend": meta["legend"],
                "probabilities": {str(index): round_probability(value) for index, value in enumerate(probs)},
                "confidence": round_probability(score_confidence(probs)),
            }
    return answers


def output_tokens(tokenizer: Any, answers: dict[str, Any]) -> int:
    return len(tokenizer(json.dumps(answers), add_special_tokens=False).input_ids)


def load_labelled_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "state" not in record or not isinstance(record.get("questions"), dict) or not record["questions"]:
                raise ValueError(f"{path}:{line_number}: record needs state and non-empty questions")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


__all__ = [
    "MAX_OPTIONS",
    "Choice",
    "JSONContent",
    "Noul",
    "Question",
    "Score",
    "SystemOneRequest",
    "SystemOneResponse",
    "choice_confidence",
    "date_facts",
    "load_labelled_records",
    "materialize_labelled",
    "option_text",
    "output_tokens",
    "question_keys",
    "render",
    "round_probability",
    "score_confidence",
    "to_answers",
    "to_record",
    "with_date_facts",
]
