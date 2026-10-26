#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"



PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi
DEBATE_LIMIT="${ENGLISH_DEBATE_LIMIT:-0}"
OUTPUT_PATH="${ENGLISH_DEBATE_OUTPUT:-artifacts/english_council_gpt_image_specific_300_v1_debate.jsonl}"
LOCK_PATH="${ROOT_DIR}/.english_gpt_debate.lock"

# Keep the lock path independent of the output file so deleting the JSONL cannot
# let a second run acquire a different lock while the first run is still active.
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "Another English debate run is already using $OUTPUT_PATH" >&2
  exit 3
fi


exec "$PYTHON" Code/meme_persona_pipeline.py \
  --backend openai \
  --model gpt-4.1-mini \
  --language English \
  --split train \
  --limit "$DEBATE_LIMIT" \
  --images-zip data/raw/English/Train/Train_images \
  --labels-xlsx data/raw/English/Train/Train_labels.xlsx \
  --persona-pool artifacts/persona_pool_en.json \
  --max-personas 300 \
  --candidate-count 6 \
  --output "$OUTPUT_PATH"
