#!/usr/bin/env python3
"""Compare the base VLM and a GRPO adapter on test memes with training-style RCR context."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from meme_pipeline.training.data import LABELS, make_prompt, read_trace_files
from train_grpo_vlm import DEFAULT_MODEL, predict_rows

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
ID_COLUMNS = ("id", "image_id")
LABEL_COLUMNS = ("label", "class")


def load_test_labels(data_root: Path, language: str) -> dict[int, str]:
    import pandas as pd

    path = data_root / language / "Test_labels.xlsx"
    if not path.is_file():
        raise FileNotFoundError(f"Test label workbook not found: {path}")
    frame = pd.read_excel(path).dropna(how="all")
    columns = {str(column).strip().lower(): column for column in frame.columns}
    id_column = next((columns[key] for key in ID_COLUMNS if key in columns), None)
    label_column = next((columns[key] for key in LABEL_COLUMNS if key in columns), None)
    if id_column is None or label_column is None:
        raise ValueError(
            f"{path} needs an ID column ({', '.join(ID_COLUMNS)}) and a label column "
            f"({', '.join(LABEL_COLUMNS)}); found {list(frame.columns)}"
        )

    labels: dict[int, str] = {}
    for _, item in frame.dropna(subset=[id_column, label_column]).iterrows():
        image_id = int(item[id_column])
        label = str(item[label_column]).strip()
        label = {"Homophobia": "Homophobic", "Transphobia": "Transphobic"}.get(label, label)
        if label not in LABELS:
            raise ValueError(f"Unexpected test label {label!r} for image {image_id}")
        if image_id in labels and labels[image_id] != label:
            raise ValueError(f"Conflicting test labels for image {image_id}")
        labels[image_id] = label
    return labels


def load_test_rows(
    trace_paths: list[str], data_root: Path, language: str
) -> list[dict[str, Any]]:
    labels = load_test_labels(data_root, language)
    image_root = data_root / language / "Test"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {image_root}")
    images: dict[int, Path] = {}
    for path in image_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = re.search(r"\d+", path.stem)
        if not match:
            continue
        image_id = int(match.group(0))
        if image_id in images:
            raise ValueError(f"Multiple test images resolve to ID {image_id}")
        images[image_id] = path

    records = read_trace_files(trace_paths)
    rows, seen = [], set()
    for record in records:
        if record.get("split") != "test":
            raise ValueError("Test evaluation accepts only traces marked split='test'.")
        if record.get("language") != language:
            raise ValueError(
                f"Trace language {record.get('language')!r} does not match {language!r}."
            )
        try:
            image_id = int(record["image_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid/missing image_id in test trace: {record.get('image_id')!r}") from exc
        if image_id in seen:
            raise ValueError(f"Duplicate test trace for image {image_id}")
        seen.add(image_id)
        if image_id not in labels:
            raise ValueError(f"Image {image_id} is absent from the test label workbook")
        if image_id not in images:
            raise ValueError(f"No test image found for trace image {image_id}")
        gold_label = labels[image_id]
        if record.get("gold_label") != gold_label:
            raise ValueError(
                f"Trace/workbook label mismatch for image {image_id}: "
                f"{record.get('gold_label')!r} != {gold_label!r}"
            )
        rows.append({
            "image_id": image_id,
            "image_path": str(images[image_id]),
            "language": language,
            "gold_label": gold_label,
            # This is the same prompt builder used to materialize GRPO train rows.
            # It deliberately omits record.gold_label and other top-level outcomes.
            "prompt": make_prompt(record),
        })

    if not rows:
        raise ValueError("No usable test traces found.")
    if seen != set(labels):
        missing = sorted(set(labels) - seen)
        extra = sorted(seen - set(labels))
        raise ValueError(
            f"Test traces must cover the labeled test set exactly. Missing IDs: {missing[:10]}; "
            f"unexpected IDs: {extra[:10]}"
        )
    return sorted(rows, key=lambda row: row["image_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True,
                        help="Full test-split RCR trace JSONL file(s), created with --split test")
    parser.add_argument("--language", choices=["Chinese", "English", "Hindi"], required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, required=True,
                        help="Existing GRPO adapter directory")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.adapter_path.is_dir():
        raise SystemExit(f"GRPO adapter directory not found: {args.adapter_path}")
    rows = load_test_rows(args.traces, args.data_root, args.language)
    print(
        f"Evaluating base and GRPO models on {len(rows)} {args.language} test memes "
        "with the training-style image + language + RCR prompt."
    )
    base = predict_rows(args.model, None, rows)
    adapted = predict_rows(args.model, str(args.adapter_path), rows)
    result = {
        "language": args.language,
        "split": "test",
        "input_mode": "image_plus_language_and_rcr_context",
        "n": len(rows),
        "base_model": args.model,
        "adapter_path": str(args.adapter_path),
        "base": base,
        "grpo_adapter": adapted,
        "note": "Both models use identical prompts. Test metrics are for reporting, not model selection.",
    }
    output_path = args.output or (
        PROJECT_ROOT / "artifacts" / f"grpo_context_test_{args.language.lower()}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "base_accuracy": base["accuracy"],
        "base_macro_f1": base["macro_f1"],
        "adapter_accuracy": adapted["accuracy"],
        "adapter_macro_f1": adapted["macro_f1"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
