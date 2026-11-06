"""In-process and HTTP SDK compatible with TypeSafe's System One call shape."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import Any, Self

import torch

from lmml.core.llm.kev.api import Choice, Noul, Score, SystemOneRequest, to_answers, to_record
from lmml.core.llm.kev.checkpoint import load_kev_model
from lmml.core.llm.kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, encode


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


class KevClient:
    """Call a local Kev checkpoint or a System One-compatible HTTP endpoint.

    ``checkpoint_path`` selects in-process inference.  A non-local ``base_url``
    selects HTTP.  The official ``typesafe_sdk`` client works against the HTTP
    server exposed by :mod:`lmml.core.llm.kev.serve`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "kev-latest",
        checkpoint_path: str | Path | None = None,
        base_model: str | Path | None = None,
        device: str = "auto",
        torch_dtype: str = "float32",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else None
        self.model = model
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.base_model = str(base_model) if base_model else None
        self.device = _resolve_device(device)
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None
        if self.checkpoint_path:
            self._load_local()
        elif not self.base_url:
            raise ValueError("KevClient needs checkpoint_path or base_url")

    def _load_local(self) -> None:
        assert self.checkpoint_path is not None
        dtype = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }.get(str(self.torch_dtype).lower(), torch.float32)
        self._model, self._tokenizer = load_kev_model(
            self.checkpoint_path,
            device=self.device,
            torch_dtype=dtype,
            base_model=self.base_model,
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
        self._tokenizer = None

    def _local_system_one(self, request: SystemOneRequest) -> SimpleNamespace:
        assert self._model is not None and self._tokenizer is not None
        record, metadata = to_record(request)
        for question in record["questions"]:
            question.setdefault("label", 0)
        encoded = encode(
            self._tokenizer,
            record,
            max_state=SERVE_MAX_STATE,
            max_branch=SERVE_MAX_BRANCH,
            strict=True,
        )
        probabilities = self._model.probabilities([encoded])[0]
        values = [[float(value) for value in row.tolist()] for row in probabilities]
        answers = to_answers(values, metadata)
        response = _namespace_answers(answers)
        response.model = request.model
        response.usage = SimpleNamespace(input_tokens=len(encoded["ids"]), output_tokens=0)
        response.latency_ms = 0.0
        response.request_id = "local"
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
        endpoint = self.base_url
        if not endpoint.endswith("/v1/systemone"):
            endpoint = f"{endpoint}/v1/systemone"
        http_request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(http_request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
            request_id = response.headers.get("x-typesafe-request-id", "")
        parsed = _namespace_answers(body["answers"])
        parsed.model = body.get("model", request.model)
        parsed.usage = SimpleNamespace(**body.get("usage", {}))
        parsed.latency_ms = float(body.get("latency_ms", 0.0))
        parsed.request_id = request_id
        parsed.truncated = body.get("truncated")
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


TypeSafeClient = KevClient


__all__ = ["Choice", "KevClient", "Noul", "Score", "TypeSafeClient"]
