"""Optional FastAPI server exposing a Kev checkpoint at POST /v1/systemone."""

from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path
from typing import Any

from lmml.core.llm.kev.api import SystemOneRequest, output_tokens, to_answers, to_record
from lmml.core.llm.kev.checkpoint import load_kev_model
from lmml.core.llm.kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, encode


class KevRuntime:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        torch_dtype: str = "float32",
        base_model: str | Path | None = None,
    ) -> None:
        self.model, self.tokenizer = load_kev_model(
            checkpoint_path,
            device=device,
            torch_dtype=torch_dtype,
            base_model=base_model,
        )
        self.model.eval()

    def answer(self, request: SystemOneRequest) -> dict[str, Any]:
        started = time.perf_counter()
        record, metadata = to_record(request)
        for question in record["questions"]:
            question.setdefault("label", 0)
        encoded = encode(
            self.tokenizer,
            record,
            max_state=SERVE_MAX_STATE,
            max_branch=SERVE_MAX_BRANCH,
            strict=True,
        )
        probabilities = self.model.probabilities([encoded])[0]
        values = [[float(value) for value in row.tolist()] for row in probabilities]
        answers = to_answers(values, metadata)
        return {
            "model": request.model,
            "answers": answers,
            "usage": {
                "input_tokens": len(encoded["ids"]),
                "output_tokens": output_tokens(self.tokenizer, answers),
            },
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def create_app(runtime: KevRuntime, api_key: str | None = None):
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
    except ImportError as error:
        raise RuntimeError(
            "Kev HTTP serving requires the optional dependencies: pip install 'lmml[serve]'"
        ) from error

    app = FastAPI(title="lmml Kev System One", version="1.0")

    @app.middleware("http")
    async def request_id_and_auth(request: Request, call_next: Any):
        if request.url.path.startswith("/v1/") and api_key:
            authorization = request.headers.get("authorization", "")
            if authorization != f"Bearer {api_key}":
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
        request_id = request.headers.get("x-typesafe-request-id") or uuid.uuid4().hex
        try:
            response = await call_next(request)
        except Exception as error:  # noqa: BLE001 - preserve the request id on middleware failures
            response = JSONResponse({"detail": str(error)}, status_code=500)
        response.headers["x-typesafe-request-id"] = request_id
        return response

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "models": [
                {
                    "name": "kev-latest",
                    "description": "lmml Kev decision model",
                    "release_date": "2026-01-01",
                    "max_state_tokens": 65536,
                },
                {
                    "name": "jev-latest",
                    "description": "TypeSafe-compatible alias",
                    "release_date": "2026-01-01",
                    "max_state_tokens": 65536,
                },
            ]
        }

    @app.post("/v1/systemone")
    def system_one(request: SystemOneRequest) -> dict[str, Any]:
        try:
            return runtime.answer(request)
        except Exception as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a Kev checkpoint with a TypeSafe-compatible API")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--base-model", default=None, help="local override for a LoRA checkpoint's base")
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit("pip install 'lmml[serve]' to use the HTTP server") from error
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    app = create_app(
        KevRuntime(
            args.checkpoint,
            device=device,
            torch_dtype=args.torch_dtype,
            base_model=args.base_model,
        ),
        api_key=os.environ.get("KEV_API_KEY"),
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


__all__ = ["KevRuntime", "create_app", "main"]
