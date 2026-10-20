#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Paste your key between these quotes. Keep this script private.
OPENAI_API_KEY="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"
export OPENAI_API_KEY

if [[ -z "$OPENAI_API_KEY" || "$OPENAI_API_KEY" == "paste-your-openai-api-key-here" ]]; then
  echo "Replace OPENAI_API_KEY in this script with your OpenAI API key before running it." >&2
  exit 2
fi

.venv/bin/pip install -r requirements.txt

exec .venv/bin/python Code/meme_persona_pipeline.py \
  --backend openai \
  --model gpt-4.1-mini \
  --language Chinese \
  --split train \
  --limit 10 \
  --images-zip data/raw/Chinese/Train/Train_images \
  --labels-xlsx data/raw/Chinese/Train/Train_labels.xlsx \
  --persona-pool artifacts/persona_council_zh.json \
  --output artifacts/chinese_council_gpt_debate.jsonl
