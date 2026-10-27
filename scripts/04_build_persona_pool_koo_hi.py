#!/usr/bin/env python3
"""Build a small Hindi persona pilot from the Koo posts archive.

The posts are grouped by Koo creator ID, but the ID is hashed before writing
the output. Only posts with clear Devanagari content are retained because the
archive's ``hi`` partition includes some posts in other languages. Archetypes
are assigned debate lenses for simulation; they are not inferred beliefs.

The JSON array is streamed directly from the ZIP so the 27 GB member does not
need to be extracted. The candidate-user hash sample bounds memory use during
the scan; it is a pilot sample, not a representative sample of Hindi users.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import time
import zipfile
from pathlib import Path
from functools import lru_cache
from typing import BinaryIO, Iterator


ARCHETYPES = [
    "Supportive / Pro-LGBTQ+ Ally",
    "Critical / Conservative",
    "Cultural / Traditional Norms",
    "Skeptical / Ambiguous Stance",
    "Educator / Factual Explainer",
    "Neutral Observer",
]
DEVANAGARI = re.compile(r"[\u0900-\u097f]")
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_OR_HANDLE = re.compile(r"(?:https?://|www\.|@[A-Z0-9_]{2,})\S*", re.IGNORECASE)
PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d(?:[\s-]?\d){8}(?!\d)")
HINDI_MARKERS = {
    "है", "हैं", "था", "थी", "थे", "नहीं", "क्या", "मुझे", "हम", "आप",
    "मैं", "मेरा", "मेरी", "और", "लेकिन", "क्योंकि", "बहुत", "यह", "वह",
    "इस", "उस", "में", "को", "से", "पर", "लिए", "करता", "करती", "करते",
    "रहा", "रही", "रहे", "होगा", "होगी", "होंगे", "अपना", "अपनी", "अपने",
}
MARATHI_MARKERS = {"आहे", "आहेत", "नाही", "नाहीत", "आणि", "मला", "तुम्ही", "आम्ही", "साठी", "करून", "पण"}


def iter_json_array(stream: io.TextIOBase, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Yield objects from a top-level JSON array without loading it all at once."""
    decoder = json.JSONDecoder()
    buffer = stream.read(chunk_size)
    pos = 0
    started = False
    eof = not buffer

    while True:
        while pos < len(buffer) and buffer[pos].isspace():
            pos += 1

        if not started:
            if pos >= len(buffer):
                if eof:
                    raise ValueError("Empty JSON input")
                buffer = stream.read(chunk_size)
                pos = 0
                eof = not buffer
                continue
            if buffer[pos] != "[":
                raise ValueError("Expected a top-level JSON array")
            started = True
            pos += 1
            continue

        while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ","):
            pos += 1

        if pos >= len(buffer):
            if eof:
                return
            buffer = buffer[pos:] + stream.read(chunk_size)
            pos = 0
            eof = not buffer
            continue

        if buffer[pos] == "]":
            return

        try:
            item, end = decoder.raw_decode(buffer, pos)
        except json.JSONDecodeError:
            if eof:
                raise
            buffer = buffer[pos:] + stream.read(chunk_size)
            pos = 0
            eof = not buffer
            continue

        pos = end
        if isinstance(item, dict):
            yield item

        # Drop consumed text periodically so the buffer remains bounded.
        if pos >= chunk_size:
            buffer = buffer[pos:]
            pos = 0


def normalize_post(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    if len(text) < 12:
        return ""
    # Avoid copying contact information or account references into persona prompts.
    if EMAIL.search(text) or URL_OR_HANDLE.search(text) or PHONE.search(text) or "@" in text:
        return ""
    letters = [char for char in text if char.isalpha()]
    devanagari_count = sum(bool(DEVANAGARI.fullmatch(char)) for char in text)
    # Requiring both script presence and a modest share of letters filters
    # obvious non-Hindi records while allowing Hindi-English code mixing.
    if devanagari_count < 5 or devanagari_count / max(len(letters), 1) < 0.20:
        return ""
    tokens = set(re.findall(r"[\u0900-\u097f]+", text))
    # The Koo Hindi partition is not language-pure. Keep only records with
    # Hindi lexical evidence and discard clear Marathi markers; this remains
    # a conservative heuristic, so the pilot still requires language review.
    if tokens & MARATHI_MARKERS or not tokens & HINDI_MARKERS:
        return ""
    return text[:2000]


def writing_habits(posts: list[str]) -> dict:
    avg_chars = sum(len(post) for post in posts) / max(len(posts), 1)
    emphatic = sum(bool(re.search(r"[!?！？]{2,}|\.{3,}|…{2,}", post)) for post in posts) / max(len(posts), 1)
    emoji = sum(bool(EMOJI.search(post)) for post in posts) / max(len(posts), 1)
    traits = ["usually concise" if avg_chars < 80 else "often writes longer posts"]
    if emphatic > 0.25:
        traits.append("uses emphatic punctuation")
    if emoji > 0.20:
        traits.append("sometimes uses emoji")
    return {
        "summary": "; ".join(traits) + ".",
        "formality": "Colloquial",
        "avg_unit_length": round(avg_chars, 1),
        "unit": "characters",
        "uses_emoji": emoji > 0.20,
        "emphatic_punctuation": emphatic > 0.25,
    }


@lru_cache(maxsize=None)
def stable_fraction(user_id: str) -> float:
    digest = hashlib.sha256(user_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1)


def build_pool(
    archive: Path,
    pool_size: int,
    min_posts: int,
    posts_per_persona: int,
    candidate_rate: float,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    states: dict[str, dict] = {}
    scan_stats = {"records_scanned": 0, "posts_with_devanagari": 0, "candidate_creators": 0}
    scan_started = time.monotonic()
    member = "Posts/hi_posts.json"

    with zipfile.ZipFile(archive) as zipped:
        try:
            raw = zipped.open(member)
        except KeyError as exc:
            raise FileNotFoundError(f"{member} not found in {archive}") from exc
        with raw, io.TextIOWrapper(raw, encoding="utf-8") as stream:
            for record in iter_json_array(stream):
                scan_stats["records_scanned"] += 1
                if scan_stats["records_scanned"] % 5_000_000 == 0:
                    elapsed = (time.monotonic() - scan_started) / 60
                    print(
                        f"Scanned {scan_stats['records_scanned']:,} Koo Hindi-locale records "
                        f"({elapsed:.1f} min)",
                        flush=True,
                    )
                user_id = record.get("creatorId")
                if not isinstance(user_id, str) or not user_id:
                    continue
                if stable_fraction(user_id) >= candidate_rate:
                    continue
                text = normalize_post(record.get("title"))
                if not text:
                    continue
                scan_stats["posts_with_devanagari"] += 1

                state = states.get(user_id)
                if state is None:
                    state = {"count": 0, "posts": [], "rng": random.Random(seed ^ int(hashlib.sha256(user_id.encode()).hexdigest()[:16], 16))}
                    states[user_id] = state
                state["count"] += 1
                post = (record.get("createdAt", 0), text)
                examples = state["posts"]
                if len(examples) < posts_per_persona:
                    examples.append(post)
                else:
                    replacement = state["rng"].randrange(state["count"])
                    if replacement < posts_per_persona:
                        examples[replacement] = post

    eligible = [(uid, state) for uid, state in states.items() if state["count"] >= min_posts]
    scan_stats["candidate_creators"] = len(states)
    scan_stats["eligible_candidate_creators"] = len(eligible)
    eligible.sort(key=lambda pair: hashlib.sha256(f"{seed}:{pair[0]}".encode()).hexdigest())
    chosen = eligible[:pool_size]
    if len(chosen) < pool_size:
        raise RuntimeError(
            f"Found {len(chosen)} eligible sampled creators; need {pool_size}. "
            "Rerun with a larger --candidate-rate or smaller --pool-size."
        )

    pool = []
    for index, (uid, state) in enumerate(chosen, start=1):
        examples = [text for _, text in sorted(state["posts"], key=lambda pair: pair[0])]
        pool.append({
            "persona_id": f"P_HI_KOO_PILOT_{index:03d}",
            "source_user_hash": hashlib.sha256(uid.encode("utf-8")).hexdigest()[:12],
            "language": "Hindi",
            "archetype": ARCHETYPES[(index - 1) % len(ARCHETYPES)],
            "profile_bio": "",
            "commenting_habits": writing_habits(examples),
            "sample_historical_posts": examples,
            "total_available_posts": state["count"],
            # Keep the legacy keys because existing debate prompts read them.
            "sample_historical_comments": examples,
            "total_available_comments": state["count"],
            "provenance": "Derived from Koo Hindi-locale posts using Devanagari and Hindi lexical heuristics; contact-bearing posts excluded; creator ID hashed. Pilot sample.",
            "persona_note": "The debate archetype is an assigned simulation lens, not an inferred belief or identity of the source author. Language filtering is heuristic and needs review; posts are historical Koo content, and the pilot is not representative of Hindi users.",
        })
    return pool, scan_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/external/Koo/koo_posts.zip?download=1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/persona_pool_hi_koo_pilot_50.json"))
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--min-posts", type=int, default=5)
    parser.add_argument("--posts-per-persona", type=int, default=6)
    parser.add_argument("--candidate-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not 0 < args.candidate_rate <= 1:
        parser.error("--candidate-rate must be in (0, 1]")
    if args.min_posts < 1 or args.posts_per_persona < 1 or args.pool_size < 1:
        parser.error("pool size and post counts must be positive")
    if not args.archive.is_file():
        parser.error(f"Archive not found: {args.archive}")

    pool, stats = build_pool(
        args.archive, args.pool_size, args.min_posts,
        args.posts_per_persona, args.candidate_rate, args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Scanned {stats['records_scanned']:,} records in Posts/hi_posts.json")
    print(f"Retained {stats['posts_with_devanagari']:,} posts passing the language/privacy heuristics")
    print(f"Sampled {stats['candidate_creators']:,} candidate creators")
    print(f"Found {stats['eligible_candidate_creators']:,} sampled creators with enough posts")
    print(f"Saved {len(pool)} personas to {args.output}")


if __name__ == "__main__":
    main()
