"""Build and split image/debate rows for task-grounded GRPO training."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LABELS = {"Homophobic", "Transphobic", "Non_Anti_LGBT"}
LANGUAGES = ("Chinese", "English", "Hindi")


def read_trace_files(paths: list[str | Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                if isinstance(row, dict):
                    records.append(row)
    return records


def build_debate_context(record: dict[str, Any]) -> str:
    analysis = record.get("meme_analysis", {})
    initial = record.get("initial_debate", [])
    refined = record.get("rcr_refined_responses", [])
    compact = []
    for initial_item, refined_item in zip(initial, refined):
        compact.append({
            "persona_id": initial_item.get("persona_id"),
            "initial_label": initial_item.get("label"),
            "initial_evidence": initial_item.get("evidence"),
            "initial_rationale": initial_item.get("rationale"),
            "self_critique": refined_item.get("self_critique"),
            "peer_critiques": refined_item.get("peer_critiques"),
            "refined_label": refined_item.get("label"),
            "refined_evidence": refined_item.get("evidence"),
            "refinement_reason": refined_item.get("refinement_reason"),
        })
    return json.dumps({"meme_analysis": analysis, "rcr_debate": compact}, ensure_ascii=False)


def make_prompt(record: dict[str, Any]) -> list[dict[str, str]]:
    lang = record.get("language", "Unknown")
    return [
        {
            "role": "system",
            "content": (
                "Classify the supplied meme as Homophobic, Transphobic, or Non_Anti_LGBT. "
                "Use visible meme evidence and the supplied three-persona RCR trace. "
                "Do not infer identity from appearance. Return only JSON with keys label, "
                "confidence, evidence, and rationale."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Language: {lang}\nMeme evidence and RCR reasoning:\n"
                f"{build_debate_context(record)}"
            ),
        },
    ]


def load_training_rows(
    records: list[dict[str, Any]],
    data_root: str | Path,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Resolve train trace IDs to train labels/images; reject test or mismatched traces."""
    import pandas as pd

    root = Path(data_root)
    image_index: dict[str, dict[int, Path]] = {}
    label_maps: dict[str, dict[int, str]] = {}
    rows = []
    seen: set[tuple[str, int]] = set()
    for language in LANGUAGES:
        label_path = root / language / "Train" / "Train_labels.xlsx"
        if not label_path.exists():
            continue
        df = pd.read_excel(label_path)
        if not {"id", "label"}.issubset(df.columns):
            raise ValueError(f"{label_path} must contain id and label columns")
        label_maps[language] = {
            int(row["id"]): str(row["label"]).strip()
            for _, row in df.iterrows()
            if not pd.isna(row["id"]) and not pd.isna(row["label"])
        }
        images: dict[int, Path] = {}
        for path in (root / language / "Train").rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                continue
            match = re.search(r"\d+", path.stem)
            if match:
                images[int(match.group(0))] = path
        image_index[language] = images

    for record in records:
        language = record.get("language")
        image_id = record.get("image_id")
        if record.get("split") != "train":
            raise ValueError("GRPO accepts only traces marked split='train'.")
        if language not in label_maps:
            raise ValueError(f"No training labels found for trace language {language!r}.")
        try:
            image_id = int(image_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid image_id in trace: {image_id!r}") from exc
        key = (language, image_id)
        if key in seen:
            continue
        seen.add(key)
        expected = label_maps[language].get(image_id)
        if expected is None:
            raise ValueError(f"Image {image_id} is not in the {language} training labels.")
        if record.get("gold_label") is not None and record.get("gold_label") != expected:
            raise ValueError(
                f"Trace label mismatch for {language} image {image_id}: "
                f"trace={record.get('gold_label')!r}, train workbook={expected!r}"
            )
        if expected not in LABELS:
            raise ValueError(f"Unexpected training label {expected!r}")
        image_path = image_index.get(language, {}).get(image_id)
        if image_path is None:
            raise ValueError(f"No training image found for {language} image {image_id}.")
        rows.append({
            "image_path": str(image_path),
            "image_id": image_id,
            "language": language,
            "gold_label": expected,
            "consensus_label": record.get("consensus_label"),
            "prompt": make_prompt(record),
        })
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError("No usable training traces found.")
    return rows


def split_train_validation(
    rows: list[dict[str, Any]], validation_fraction: float = 0.1, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["language"]].append(row)
    train, validation = [], []
    for language in sorted(groups):
        group = list(groups[language])
        random.Random(seed + sum(map(ord, language))).shuffle(group)
        if len(group) < 2:
            train.extend(group)
            continue
        n_validation = max(1, round(len(group) * validation_fraction))
        if n_validation >= len(group):
            n_validation = len(group) - 1
        counts = Counter(row["gold_label"] for row in group)
        stratify_possible = (
            n_validation >= len(counts)
            and len(group) - n_validation >= len(counts)
            and min(counts.values()) >= 2
        )
        if stratify_possible:
            by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in group:
                by_label[row["gold_label"]].append(row)
            chosen_validation = []
            for label in sorted(by_label):
                members = by_label[label]
                count = max(1, round(len(members) * validation_fraction))
                chosen_validation.extend(members[:count])
            validation.extend(chosen_validation)
            chosen_ids = {id(row) for row in chosen_validation}
            train.extend(row for row in group if id(row) not in chosen_ids)
        else:
            validation.extend(group[:n_validation])
            train.extend(group[n_validation:])
    if not train:
        raise ValueError("Training split is empty after validation split.")
    return train, validation
