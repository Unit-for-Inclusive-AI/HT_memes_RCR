#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

OUTPUT_PATH="${HINDI_DEBATE_OUTPUT:-artifacts/hindi_koo_persona_gpt_full_debate.jsonl}"
LOCK_PATH="${ROOT_DIR}/.hindi_gpt_debate.lock"

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "Another Hindi debate run is already using $OUTPUT_PATH" >&2
  exit 3
fi

exec "$PYTHON" Code/meme_persona_pipeline.py \
  --backend openai \
  --model gpt-4.1-mini \
  --language Hindi \
  --split train \
  --limit 0 \
  --images-zip data/raw/Hindi/Train/Train_Images \
  --labels-xlsx data/raw/Hindi/Train/Train_labels.xlsx \
  --persona-pool artifacts/persona_pool_hi_koo_provisional_300.json \
  --max-personas 300 \
  --candidate-count 6 \
  --output "$OUTPUT_PATH"
