"""In-process and HTTP TypeSafeClient compatible with Jev (Qwen3.8-27B + LoRA + Scalar Head) checkpoints."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import Any, Self

import torch

from lmml.core.llm.jev.api import (
    Choice,
    Noul,
    Score,
    SystemOneRequest,
    compile_request,
    format_response,
)
from lmml.core.llm.jev.metrics import softmax
from lmml.core.llm.jev.model import JevDecisionModel


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _namespace_answers(answers: dict[str, Any]) -> SimpleNamespace:
    values = {key: SimpleNamespace(**value) for key, value in answers.items()}
    return SimpleNamespace(
        answers=values,
        nouls={key: value for key, value in values.items() if value.type == "noul"},
        choices={key: value for key, value in values.items() if value.type == "choice"},
        scores={key: value for key, value in values.items() if value.type == "score"},
    )


class JevClient:
    """Call a local Jev checkpoint (`adapter/` + `head.pt` + `temperature.json`) or HTTP `/v1/systemone`."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "typesafe/jev-1.14",
        checkpoint_path: str | Path | None = None,
        base_model: str | Path | None = None,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        max_length: int = 512,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else None
        self.model = model
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.base_model = str(base_model) if base_model else None
        self.device = _resolve_device(device)
        self.torch_dtype = torch_dtype
        self.max_length = int(max_length)
        self._model: JevDecisionModel | None = None
        if self.checkpoint_path:
            self._load_local()
        elif not self.base_url:
            raise ValueError("JevClient needs checkpoint_path or base_url")

    def _load_local(self) -> None:
        assert self.checkpoint_path is not None
        dtype = (
            "float32"
            if self.device == "cpu" and str(self.torch_dtype).lower() in ("auto", "float32", "fp32")
            else self.torch_dtype
        )
        self._model = JevDecisionModel.load(
            self.checkpoint_path,
            base_model_path=self.base_model,
            device=self.device,
            torch_dtype=dtype,
            is_trainable=False,
            max_length=self.max_length,
            strict_length=False,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._model = None

    def _local_system_one(self, request: SystemOneRequest) -> SimpleNamespace:
        assert self._model is not None
        start = time.perf_counter()
        raw_questions = {
            key: question.model_dump(exclude_none=False)
            for key, question in request.questions.items()
        }
        records = compile_request(request.state, raw_questions)
        with torch.inference_mode():
            logits = self._model(records)
        probs = [
            softmax(row.detach().float().cpu().tolist(), temperature=self._model.temperature)
            for row in logits
        ]
        formatted = format_response(records, probs)
        response = _namespace_answers(formatted["answers"])
        response.model = request.model
        response.usage = SimpleNamespace(
            input_tokens=int(self._model.last_input_tokens),
            output_tokens=0,
        )
        response.latency_ms = (time.perf_counter() - start) * 1000.0
        response.temperature = float(self._model.temperature)
        return response

    def _http_system_one(self, request: SystemOneRequest) -> SimpleNamespace:
        assert self.base_url is not None
        payload = json.dumps(
            {
                "state": request.state,
                "model": request.model,
                "questions": {
                    key: question.model_dump(exclude_none=True)
                    for key, question in request.questions.items()
                },
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = (
            self.base_url
            if self.base_url.endswith("/v1/systemone")
            else f"{self.base_url}/v1/systemone"
        )
        http_request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(http_request, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        parsed = _namespace_answers(body["answers"])
        parsed.model = body.get("model", request.model)
        parsed.usage = SimpleNamespace(**body.get("usage", {}))
        parsed.latency_ms = float(body.get("latency_ms", 0.0))
        return parsed

    def system_one(
        self,
        *,
        state: Any,
        questions: dict[str, Any],
        model: str | None = None,
    ) -> SimpleNamespace:
        request = SystemOneRequest.model_validate(
            {"state": state, "questions": questions, "model": model or self.model}
        )
        return self._local_system_one(request) if self._model is not None else self._http_system_one(request)


TypeSafeClient = JevClient

__all__ = ["Choice", "JevClient", "Noul", "Score", "TypeSafeClient"]
