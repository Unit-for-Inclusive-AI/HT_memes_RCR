#!/usr/bin/env python3
"""Build an English persona pool from the PAN19 author profiling training split.

Only authors listed as human in truth-train.txt are eligible. Persona text is
derived from their English XML documents; PAN19 labels are used only to filter
bot accounts and are not copied into the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ARCHETYPES = [
    "Supportive / Pro-LGBTQ+ Ally",
    "Critical / Conservative",
    "Cultural / Traditional Norms",
    "Skeptical / Ambiguous Stance",
    "Educator / Factual Explainer",
    "Neutral Observer",
]


def read_human_train_ids(labels_path: Path) -> list[str]:
    human_ids = []
    for line_no, line in enumerate(labels_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(":::")]
        if len(parts) < 2:
            raise ValueError(f"Malformed label at {labels_path}:{line_no}")
        if parts[1].lower() == "human":
            human_ids.append(parts[0])
    if len(human_ids) != len(set(human_ids)):
        raise ValueError(f"Duplicate author IDs in {labels_path}")
    return human_ids


def load_english_posts(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    if root.attrib.get("lang", "").lower() != "en":
        raise ValueError(f"Unexpected author language in {xml_path}")
    posts = []
    for node in root.findall("./documents/document"):
        text = " ".join((node.text or "").split())
        # Ignore URLs, very short fragments, and text without an English-script word.
        words = [w for w in text.split() if not w.startswith(("http://", "https://", "@"))]
        if len(words) < 4 or not re.search(r"[A-Za-z]{3,}", text):
            continue
        if re.search(r"[\u4e00-\u9fff\u0900-\u097f]", text):
            continue
        posts.append(text[:2000])
    return posts


def writing_habits(posts: list[str]) -> dict:
    average_words = sum(len(post.split()) for post in posts) / max(len(posts), 1)
    emphatic = sum(bool(re.search(r"[!?]{2,}|\.{3,}", post)) for post in posts) / max(len(posts), 1)
    emoji = sum(bool(re.search(r"[😀-🙏]", post)) for post in posts) / max(len(posts), 1)
    traits = ["usually concise" if average_words < 12 else "often gives detailed posts"]
    if emphatic > 0.25:
        traits.append("uses emphatic punctuation")
    if emoji > 0.20:
        traits.append("sometimes uses emoji")
    return {
        "summary": "; ".join(traits) + ".",
        "formality": "Colloquial" if average_words < 12 else "Formal",
        "avg_unit_length": round(average_words, 1),
        "unit": "words",
        "uses_emoji": emoji > 0.20,
        "emphatic_punctuation": emphatic > 0.25,
    }


def select_examples(posts: list[str], limit: int = 6) -> list[str]:
    if len(posts) <= limit:
        return posts
    # Spread examples through the available history instead of taking only its start.
    indexes = [round(i * (len(posts) - 1) / (limit - 1)) for i in range(limit)]
    return [posts[i] for i in indexes]


def build_pool(authors_dir: Path, labels_path: Path, pool_size: int, seed: int) -> list[dict]:
    human_ids = [uid for uid in read_human_train_ids(labels_path) if (authors_dir / f"{uid}.xml").is_file()]
    rng = random.Random(seed)
    rng.shuffle(human_ids)

    pool = []
    for user_id in human_ids:
        posts = load_english_posts(authors_dir / f"{user_id}.xml")
        if len(posts) < 6:
            continue
        idx = len(pool)
        pool.append({
            "persona_id": f"P_EN_{idx + 1:04d}",
            "source_user_hash": hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12],
            "language": "English",
            # These are assigned debate lenses for the simulation, not claims about the users.
            "archetype": ARCHETYPES[idx % len(ARCHETYPES)],
            "profile_bio": "",
            "commenting_habits": writing_habits(posts),
            "sample_historical_comments": select_examples(posts),
            "total_available_comments": len(posts),
            "provenance": "Derived from PAN19 English training-split human-author posts; source ID hashed.",
            "persona_note": "The debate archetype is an assigned simulation lens, not an inferred belief or identity of the source author.",
        })
        if len(pool) == pool_size:
            break

    if len(pool) < pool_size:
        raise RuntimeError(f"Only {len(pool)} eligible authors had at least six usable posts; requested {pool_size}.")
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authors-dir", type=Path, default=Path(
        "data/external/English/pan19-author-profiling-20200229/"
        "pan19-author-profiling-training-2019-02-18/en"))
    parser.add_argument("--labels", type=Path, default=Path(
        "data/external/English/pan19-author-profiling-20200229/"
        "pan19-author-profiling-training-2019-02-18/en/truth-train.txt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/persona_pool_en.json"))
    parser.add_argument("--pool-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    pool = build_pool(args.authors_dir, args.labels, args.pool_size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(pool)} English personas to {args.output}")


if __name__ == "__main__":
    main()
