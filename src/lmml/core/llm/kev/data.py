"""Kev datasets, augmentation and Lightning data modules."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset

from lmml.core.llm.kev.api import SystemOneRequest, materialize_labelled, to_record
from lmml.core.llm.kev.model import (
    MAX_BRANCH,
    MAX_STATE,
    ContextOverflow,
    encode,
)
from lmml.datas.llm.prepare import load_prepared_dataset

NONE_OPTIONS = [
    ("other", "None of the above"),
    ("other", "A reason that fits none of the above"),
    ("none", "None of these"),
    ("other", "Something else"),
    ("not_listed", "Not listed here"),
    ("none_of_the_above", None),
    ("other", "A category that fits none of the above"),
    ("other", "None of the listed options apply"),
    ("unknown", "Cannot be determined from the options given"),
    ("other", "Other"),
    ("none", None),
    ("other", "An answer not covered by the other options"),
    ("no_match", "No option matches"),
]
DISTRACTORS = {
    "weather": "Bad weather caused it",
    "purple": "The colour purple",
    "pancakes": "A recipe for pancakes",
    "taxes": "Unrelated: quarterly tax filing",
}


def augment(
    record: dict[str, Any],
    rng: random.Random,
    *,
    p_none: float = 0.1,
    p_none_distract: float = 0.12,
    p_distract: float = 0.15,
) -> dict[str, Any]:
    """Permute options and inject none-of-the-above/distractors into Choice questions."""
    if min(p_none, p_none_distract, p_distract) < 0 or p_none + p_none_distract + p_distract > 1:
        raise ValueError("augmentation probabilities must be nonnegative and sum to at most one")
    output = {"state": record["state"], "questions": {}}
    for question_id, question in record["questions"].items():
        if question.get("type") != "choice":
            output["questions"][question_id] = dict(question)
            continue
        criteria = dict(question["criteria"])
        label = question["label"]
        if question.get("target") is not None:
            keys = list(criteria)
            rng.shuffle(keys)
            output["questions"][question_id] = {
                **question,
                "criteria": {key: criteria[key] for key in keys},
            }
            continue
        draw = rng.random()
        none_options = [(key, value) for key, value in NONE_OPTIONS if key not in criteria]
        distractors = [key for key in DISTRACTORS if key not in criteria]
        if len(criteria) > 2 and draw < p_none and none_options:
            key, value = rng.choice(none_options)
            criteria.pop(label)
            criteria[key] = value
            label = key
        elif p_none <= draw < p_none + p_none_distract and len(criteria) < 255 and none_options:
            key, value = rng.choice(none_options)
            criteria[key] = value
        elif (
            p_none + p_none_distract <= draw < p_none + p_none_distract + p_distract
            and len(criteria) < 255
            and distractors
        ):
            key = rng.choice(distractors)
            criteria[key] = DISTRACTORS[key]
        keys = list(criteria)
        rng.shuffle(keys)
        output["questions"][question_id] = {
            **question,
            "criteria": {key: criteria[key] for key in keys},
            "label": label,
        }
    return output


def none_pair(record: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    """Build a minimal pair where the true option is present once and removed once."""
    eligible = [
        (question_id, question)
        for question_id, question in record["questions"].items()
        if question.get("type") == "choice"
        and len(question["criteria"]) >= 3
        and question.get("target") is None
    ]
    if not eligible:
        return []
    question_id, question = rng.choice(eligible)
    candidates = [(key, value) for key, value in NONE_OPTIONS if key not in question["criteria"]]
    key, value = rng.choice(candidates or [("none_of_these", None)])
    keys = [*question["criteria"], key]
    rng.shuffle(keys)
    present = {
        **question,
        "criteria": {item: (value if item == key else question["criteria"][item]) for item in keys},
    }
    absent = {
        **present,
        "criteria": {item: content for item, content in present["criteria"].items() if item != question["label"]},
        "label": key,
    }
    return [
        {"state": record["state"], "questions": {question_id: present}},
        {"state": record["state"], "questions": {question_id: absent}},
    ]


def _record_variants(
    record: dict[str, Any],
    index: int,
    seed: int,
    *,
    train: bool,
    p_none: float,
    p_none_distract: float,
    p_distract: float,
    p_none_pair: float,
) -> list[dict[str, Any]]:
    if not train:
        return [record]
    rng = random.Random(f"{seed}:{index}")
    variants = [
        augment(
            record,
            rng,
            p_none=p_none,
            p_none_distract=p_none_distract,
            p_distract=p_distract,
        )
    ]
    if p_none_pair > 0 and rng.random() < p_none_pair:
        variants.extend(none_pair(record, rng))
    return variants


class KevArrowDataset(Dataset):
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.dataset[int(index)])
        raw = row.get("record_json")
        if isinstance(raw, str):
            record = json.loads(raw)
        elif isinstance(raw, dict):
            record = raw
        else:
            raise TypeError(f"prepared Kev row {index} has no record_json")
        return {"index": int(index), "record": record}


class KevCollator:
    def __init__(
        self,
        tokenizer: Any,
        *,
        max_state: int = MAX_STATE,
        max_branch: int = MAX_BRANCH,
        train: bool = False,
        seed: int = 42,
        p_none: float = 0.1,
        p_none_distract: float = 0.12,
        p_distract: float = 0.15,
        p_none_pair: float = 0.25,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_state = int(max_state)
        self.max_branch = int(max_branch)
        self.train = bool(train)
        self.seed = int(seed)
        self.p_none = float(p_none)
        self.p_none_distract = float(p_none_distract)
        self.p_distract = float(p_distract)
        self.p_none_pair = float(p_none_pair)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded_records: list[dict[str, Any]] = []
        raw_records: list[dict[str, Any]] = []
        for item in batch:
            variants = _record_variants(
                item["record"],
                item["index"],
                self.seed,
                train=self.train,
                p_none=self.p_none,
                p_none_distract=self.p_none_distract,
                p_distract=self.p_distract,
                p_none_pair=self.p_none_pair,
            )
            for variant in variants:
                materialized = materialize_labelled(variant)
                try:
                    encoded = encode(
                        self.tokenizer,
                        materialized,
                        max_state=self.max_state,
                        max_branch=self.max_branch,
                        strict=True,
                    )
                except ContextOverflow:
                    continue
                encoded_records.append(encoded)
                raw_records.append(variant)
        return {"records": encoded_records, "raw_records": raw_records}


class KevDataModule(pl.LightningDataModule):
    def __init__(
        self,
        prepared_path: str | Path,
        tokenizer: Any,
        *,
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 0,
        max_state: int = MAX_STATE,
        max_branch: int = MAX_BRANCH,
        seed: int = 42,
        p_none: float = 0.1,
        p_none_distract: float = 0.12,
        p_distract: float = 0.15,
        p_none_pair: float = 0.25,
        predict_requests: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.prepared_path = Path(prepared_path)
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size)
        self.num_workers = max(0, int(num_workers))
        self.max_state = int(max_state)
        self.max_branch = int(max_branch)
        self.seed = int(seed)
        self.p_none = float(p_none)
        self.p_none_distract = float(p_none_distract)
        self.p_distract = float(p_distract)
        self.p_none_pair = float(p_none_pair)
        self.predict_requests = list(predict_requests or [])
        self.train_dataset: KevArrowDataset | None = None
        self.val_dataset: KevArrowDataset | None = None
        self.calibration_dataset: KevArrowDataset | None = None
        self.test_dataset: KevArrowDataset | None = None
        self._dataset_dict: Any = None
        self.train_collator = KevCollator(
            tokenizer,
            max_state=max_state,
            max_branch=max_branch,
            train=True,
            seed=seed,
            p_none=p_none,
            p_none_distract=p_none_distract,
            p_distract=p_distract,
            p_none_pair=p_none_pair,
        )
        self.eval_collator = KevCollator(
            tokenizer,
            max_state=max_state,
            max_branch=max_branch,
            train=False,
            seed=seed,
        )

    def setup(self, stage: str | None = None) -> None:
        if self._dataset_dict is None:
            self._dataset_dict = load_prepared_dataset(self.prepared_path)
        data = self._dataset_dict
        if self.train_dataset is None:
            self.train_dataset = KevArrowDataset(data["train"])
        if self.val_dataset is None:
            self.val_dataset = KevArrowDataset(data.get("val", data.get("calibration", data["train"])))
        if self.calibration_dataset is None:
            self.calibration_dataset = KevArrowDataset(
                data.get("calibration", data.get("val", data["train"]))
            )
        if self.test_dataset is None:
            self.test_dataset = KevArrowDataset(
                data.get("test", data.get("development", data.get("val", data["train"])))
            )

    def train_dataloader(self) -> DataLoader:
        self.setup("fit")
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_collator,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        self.setup("fit")
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.eval_collator,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        self.setup("test")
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.eval_collator,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def predict_dataloader(self) -> DataLoader:
        requests = list(self.predict_requests)
        if not requests:
            self.setup("predict")
            assert self.val_dataset is not None
            requests = [
                dict(self.val_dataset.dataset[index])
                for index in range(min(4, len(self.val_dataset)))
            ]
            requests = [json.loads(row["record_json"]) if isinstance(row.get("record_json"), str) else row for row in requests]
        encoded_requests: list[dict[str, Any]] = []
        for request in requests:
            if "record_json" in request:
                request = json.loads(request["record_json"])
            encoded_requests.append({"index": len(encoded_requests), "record": request})
        return DataLoader(
            encoded_requests,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=KevPredictionCollator(self.tokenizer, self.max_state, self.max_branch),
            num_workers=0,
        )

    def calibration_records(self) -> list[dict[str, Any]]:
        self.setup("fit")
        assert self.calibration_dataset is not None
        return [
            dict(self.calibration_dataset.dataset[index])
            for index in range(len(self.calibration_dataset))
        ]


class KevPredictionCollator:
    def __init__(self, tokenizer: Any, max_state: int = MAX_STATE, max_branch: int = MAX_BRANCH) -> None:
        self.tokenizer = tokenizer
        self.max_state = int(max_state)
        self.max_branch = int(max_branch)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        requests: list[dict[str, Any]] = []
        encodings: list[dict[str, Any]] = []
        for item in batch:
            request = item["record"]
            parsed = SystemOneRequest.model_validate(
                {"state": request["state"], "model": request.get("model", "kev-latest"), "questions": request["questions"]}
            )
            record, metadata = to_record(parsed)
            for question in record["questions"]:
                question.setdefault("label", 0)
            requests.append(request)
            encodings.append(
                {
                    "record": record,
                    "metadata": metadata,
                    "encoded": encode(
                        self.tokenizer,
                        record,
                        max_state=self.max_state,
                        max_branch=self.max_branch,
                        strict=True,
                    ),
                }
            )
        return {"requests": requests, "items": encodings}


__all__ = [
    "DISTRACTORS",
    "NONE_OPTIONS",
    "KevArrowDataset",
    "KevCollator",
    "KevDataModule",
    "KevPredictionCollator",
    "augment",
    "none_pair",
]
