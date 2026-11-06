"""Calibration and selective-prediction metrics for Kev."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

EPSILON = 1e-9
TEMPERATURE_FIT = {"aggregation": "micro", "points": 121}


def ece(confidence: Iterable[float], correct: Iterable[bool], bins: int = 10) -> float:
    confidence_array = np.asarray(list(confidence), dtype=float)
    correct_array = np.asarray(list(correct), dtype=bool)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for low, high in itertools.pairwise(edges):
        mask = (confidence_array >= low) & (
            confidence_array < high if high < 1 else confidence_array <= high
        )
        if mask.any():
            value += mask.mean() * abs(correct_array[mask].mean() - confidence_array[mask].mean())
    return float(value)


def nll(probabilities: np.ndarray, label: int, temperature: float = 1.0) -> float:
    if temperature == 1:
        return float(-math.log(max(float(probabilities[label]), EPSILON)))
    logits = np.log(np.maximum(probabilities, EPSILON)) / float(temperature)
    logits -= logits.max()
    log_normalizer = math.log(float(np.exp(logits).sum()))
    return float(log_normalizer - logits[label])


def fit_temperature(rows: list[dict[str, Any]], *, aggregation: str = "micro", points: int = 121) -> float:
    """Fit one positive temperature on raw probability rows."""
    if not rows:
        raise ValueError("cannot fit a temperature without calibration rows")
    candidates = np.exp(np.linspace(np.log(0.25), np.log(4.0), int(points)))
    if aggregation not in {"micro", "macro"}:
        raise ValueError("aggregation must be micro or macro")
    counts = defaultdict(int)
    for row in rows:
        counts[row.get("task", row.get("qtype", "all"))] += 1
    weights = np.asarray(
        [
            1.0 / counts[row.get("task", row.get("qtype", "all"))]
            if aggregation == "macro"
            else 1.0
            for row in rows
        ],
        dtype=float,
    )
    losses: list[float] = []
    for temperature in candidates:
        values = [
            nll(np.asarray(row["probabilities"], dtype=float), int(row["label"]), float(temperature))
            for row in rows
        ]
        losses.append(float(np.average(values, weights=weights)))
    return float(candidates[int(np.argmin(losses))])


def metrics(rows: list[dict[str, Any]], temperature: float = 1.0) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    probabilities = [np.asarray(row["probabilities"], dtype=float) for row in rows]
    labels = [int(row["label"]) for row in rows]
    confidence = [float(values.max()) for values in probabilities]
    correct = [int(values.argmax() == label) for values, label in zip(probabilities, labels)]
    brier = [
        float(((values - np.eye(len(values))[label]) ** 2).sum())
        for values, label in zip(probabilities, labels)
    ]
    result = {
        "n": len(rows),
        "nll": float(np.mean([nll(values, label, temperature) for values, label in zip(probabilities, labels)])),
        "acc": float(np.mean(correct)),
        "ece": ece(confidence, correct),
        "brier": float(np.mean(brier)),
        "mean_conf": float(np.mean(confidence)),
        "confident_error_rate": float(
            np.mean([conf >= 0.9 and not ok for conf, ok in zip(confidence, correct)])
        ),
        "coverage_at_5pct_error": coverage_at_error(confidence, correct, 0.05),
    }
    score_rows = [row for row in rows if row.get("qtype") == "score"]
    if score_rows:
        result["score_mae"] = float(
            np.mean(
                [
                    abs(
                        float(
                            np.asarray(row["probabilities"], dtype=float)
                            @ np.arange(len(row["probabilities"]))
                        )
                        - int(row["label"])
                    )
                    for row in score_rows
                ]
            )
        )
    return result


def coverage_at_error(confidence: Iterable[float], correct: Iterable[bool], budget: float) -> float:
    confidence_array = np.asarray(list(confidence), dtype=float)
    correct_array = np.asarray(list(correct), dtype=bool)
    if len(confidence_array) == 0:
        return 0.0
    order = np.argsort(-confidence_array, kind="stable")
    errors = 0
    best = 0.0
    for accepted, index in enumerate(order, start=1):
        errors += int(not correct_array[index])
        if errors / accepted <= budget:
            best = accepted / len(confidence_array)
    return float(best)


__all__ = ["TEMPERATURE_FIT", "coverage_at_error", "ece", "fit_temperature", "metrics", "nll"]
