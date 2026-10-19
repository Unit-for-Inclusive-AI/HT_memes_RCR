#!/usr/bin/env python3
"""CLI runner for Step 1: Persona Pool Generation."""

import argparse
import json
from pathlib import Path
import sys
import requests

# Make the src-layout package importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from meme_pipeline.personas.extractor import build_persona_pool

DEFAULT_SOME_URL = (
    "https://raw.githubusercontent.com/LivXue/SoMe/main/"
    "datasets/user_comment_silmulation/ground_truth.json"
)


def load_input_data(source_path: str):
    if source_path.startswith(("http://", "https://")):
        print(f"Downloading SoMe benchmark from {source_path}...")
        resp = requests.get(source_path, timeout=90)
        resp.raise_for_status()
        return resp.json()
    with open(source_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Step 1: Construct Persona Pool")
    parser.add_argument("--source", default=DEFAULT_SOME_URL, help="URL or path to data/external/SoMe_ground_truth.json")
    parser.add_argument("--output", default="artifacts/persona_pool.json", help="Path to save output JSON")
    parser.add_argument("--language", default="English", help="Target language filter")
    parser.add_argument("--pool-size", type=int, default=300, help="Number of personas to extract")
    args = parser.parse_args()

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    raw_data = load_input_data(args.source)
    pool = build_persona_pool(raw_data, language=args.language, max_pool_size=args.pool_size)

    out_file.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[✓] Saved {len(pool)} personas to {out_file.resolve()}")


if __name__ == "__main__":
    main()
