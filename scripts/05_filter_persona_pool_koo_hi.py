#!/usr/bin/env python3
"""Build a balanced provisional Hindi pool from sampled Koo creators.

Requires lingua-language-detector==2.2.0. The classifier is automatic; the
result remains provisional until a Hindi-speaking reviewer checks the posts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from lingua import Language, LanguageDetectorBuilder
except ImportError as exc:
    raise SystemExit(
        "Install the optional language detector with: "
        "python -m pip install -r requirements-language-detection.txt"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = Path(__file__).with_name("04_build_persona_pool_koo_hi.py")
SPEC = importlib.util.spec_from_file_location("koo_hi_builder", BUILDER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load persona builder: {BUILDER_PATH}")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)

ARCHETYPES = BUILDER.ARCHETYPES
LINGUA_VERSION = "2.2.0"


def label_post(
    text: str,
    top_detector: Any,
    segment_detector: Any,
) -> tuple[str, str, float, str, float]:
    confidence_values = top_detector.compute_language_confidence_values(text)
    top = confidence_values[0]
    second = confidence_values[1]
    top_name = top.language.name.title()
    second_name = second.language.name.title()
    segments = segment_detector.detect_multiple_languages_of(text)
    has_non_hindi_marathi_segment = any(
        segment.language not in (Language.HINDI, Language.MARATHI)
        for segment in segments
    )

    if top_name in ("Hindi", "Marathi") and has_non_hindi_marathi_segment:
        label = "Code-mixed"
    elif top.value < 0.65:
        label = "Unclear"
    elif top_name == "Marathi":
        label = "Marathi"
    elif top_name == "Hindi":
        label = "Hindi"
    else:
        label = "Unclear"
    return label, top_name, float(top.value), second_name, float(second.value)


def stable_order(persona: dict, seed: int) -> str:
    return hashlib.sha256(
        f"{seed}:{persona['source_user_hash']}".encode("utf-8")
    ).hexdigest()


def build(args: argparse.Namespace) -> tuple[list[dict], list[dict], dict]:
    candidates, scan_stats = BUILDER.build_pool(
        archive=args.archive,
        pool_size=args.candidate_size,
        min_posts=args.min_posts,
        posts_per_persona=args.posts_per_persona,
        candidate_rate=args.candidate_rate,
        seed=args.seed,
    )

    top_detector = (
        LanguageDetectorBuilder.from_languages(Language.HINDI, Language.MARATHI)
        .with_preloaded_language_models()
        .build()
    )
    segment_detector = (
        LanguageDetectorBuilder.from_all_languages()
        .with_preloaded_language_models()
        .build()
    )

    candidate_rows: list[dict] = []
    eligible: dict[str, list[tuple[str, float]]] = {}
    counts: Counter[str] = Counter()
    for persona in candidates:
        hindi_posts = []
        hindi_confidences = []
        for index, text in enumerate(persona["sample_historical_posts"], start=1):
            label, top, top_conf, second, second_conf = label_post(
                text, top_detector, segment_detector
            )
            candidate_rows.append({
                "candidate_persona_id": persona["persona_id"],
                "source_user_hash": persona["source_user_hash"],
                "sample_post_index": index,
                "post_text": text,
                "auto_language_label": label,
                "top_language": top,
                "top_confidence": top_conf,
                "second_language": second,
                "second_confidence": second_conf,
            })
            if label == "Hindi":
                hindi_posts.append(text)
                hindi_confidences.append(top_conf)
        if len(hindi_posts) >= args.min_hindi_posts:
            eligible[persona["persona_id"]] = list(zip(hindi_posts, hindi_confidences))

    by_archetype: dict[str, list[dict]] = {name: [] for name in ARCHETYPES}
    candidate_by_id = {persona["persona_id"]: persona for persona in candidates}
    for persona_id in eligible:
        persona = candidate_by_id[persona_id]
        by_archetype[persona["archetype"]].append(persona)

    deficient = {
        name: len(items)
        for name, items in by_archetype.items()
        if len(items) < args.per_archetype
    }
    if deficient:
        raise RuntimeError(
            "Not enough automatically Hindi-labeled creators for a balanced pool: "
            f"{deficient}. Increase --candidate-size and rerun."
        )

    rng = random.Random(args.seed)
    final_pool: list[dict] = []
    selected_ids: dict[str, str] = {}
    for archetype in ARCHETYPES:
        group = sorted(
            by_archetype[archetype],
            key=lambda persona: stable_order(persona, args.seed),
        )
        rng.shuffle(group)
        for persona in group[: args.per_archetype]:
            filtered_posts = eligible[persona["persona_id"]]
            final_persona = dict(persona)
            final_id = f"P_HI_KOO_{len(final_pool) + 1:04d}"
            selected_ids[persona["persona_id"]] = final_id
            posts = [text for text, _ in filtered_posts]
            confidences = [confidence for _, confidence in filtered_posts]
            final_persona.update({
                "persona_id": final_id,
                "language": "Hindi",
                "commenting_habits": BUILDER.writing_habits(posts),
                "sample_historical_posts": posts,
                "sample_historical_comments": posts,
                "sampled_hindi_post_count": len(posts),
                "provenance": (
                    "Derived from the Koo Hindi-locale partition; candidate creators "
                    "selected by deterministic 2% hash sample, then filtered with "
                    "Lingua language identification. Source creator ID hashed."
                ),
                "persona_note": (
                    "Language labels are automatic and provisional, pending Hindi-speaker "
                    "review. The debate archetype is assigned for simulation, not inferred "
                    "from the source author's beliefs or identity. Koo posts are historical; "
                    "this pool is not representative of Hindi users."
                ),
                "language_filter": {
                    "method": "Lingua Hindi/Marathi confidence plus all-language segment detection",
                    "model_version": LINGUA_VERSION,
                    "rule": (
                        "Hindi top label with confidence >= 0.65 and no detected "
                        "non-Hindi/non-Marathi segment; retain creators with at least "
                        f"{args.min_hindi_posts} retained posts"
                    ),
                    "post_confidences": confidences,
                },
            })
            final_persona.pop("total_available_posts", None)
            final_persona.pop("total_available_comments", None)
            final_pool.append(final_persona)

    label_rows = []
    final_id_lookup = {
        persona["source_user_hash"]: persona["persona_id"] for persona in final_pool
    }
    for row in candidate_rows:
        row["selected_persona_id"] = final_id_lookup.get(row["source_user_hash"], "")
        label_rows.append(row)

    stats = {
        **scan_stats,
        "candidate_personas": len(candidates),
        "candidate_posts_labeled": len(candidate_rows),
        "qualifying_creators": len(eligible),
        "qualifying_by_archetype": {
            name: len(items) for name, items in by_archetype.items()
        },
        "retained_personas": len(final_pool),
        "retained_posts": sum(len(p["sample_historical_posts"]) for p in final_pool),
        "retained_by_archetype": dict(Counter(p["archetype"] for p in final_pool)),
        "label_counts": dict(Counter(row["auto_language_label"] for row in label_rows)),
    }
    return final_pool, label_rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/external/Koo/koo_posts.zip?download=1"))
    parser.add_argument("--candidate-size", type=int, default=1000)
    parser.add_argument("--candidate-rate", type=float, default=0.02)
    parser.add_argument("--min-posts", type=int, default=5)
    parser.add_argument("--posts-per-persona", type=int, default=6)
    parser.add_argument("--min-hindi-posts", type=int, default=5)
    parser.add_argument("--per-archetype", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("artifacts/persona_pool_hi_koo_provisional_300.json"))
    parser.add_argument("--labels-output", type=Path, default=Path("artifacts/koo_hi_persona_language_auto_labels_1000.csv"))
    args = parser.parse_args()
    if args.candidate_size < args.per_archetype * len(ARCHETYPES):
        parser.error("--candidate-size must be at least six times --per-archetype")
    if not 0 < args.candidate_rate <= 1:
        parser.error("--candidate-rate must be in (0, 1]")

    pool, label_rows, stats = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "candidate_persona_id", "selected_persona_id", "source_user_hash",
        "sample_post_index", "post_text", "auto_language_label", "top_language",
        "top_confidence", "second_language", "second_confidence",
        "manual_review_label", "review_notes",
    ]
    with args.labels_output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in label_rows:
            writer.writerow({**row, "manual_review_label": "", "review_notes": ""})
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved {len(pool)} provisional personas to {args.output}")
    print(f"Saved {len(label_rows)} automatic labels to {args.labels_output}")


if __name__ == "__main__":
    main()
