#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# One-time authentication: gcloud auth application-default login
# In Google Cloud, enable Vertex AI API and grant this identity Vertex AI User access.
GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-ht-memes-kishore}"
GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
export GOOGLE_CLOUD_PROJECT GOOGLE_CLOUD_LOCATION

if [[ -z "$GOOGLE_CLOUD_PROJECT" || "$GOOGLE_CLOUD_PROJECT" == "your-gcp-project-id" ]]; then
  echo "Set GOOGLE_CLOUD_PROJECT in this script to your Google Cloud project ID." >&2
  exit 2
fi

ADC_FILE="$HOME/.config/gcloud/application_default_credentials.json"
if [[ ! -f "$ADC_FILE" && -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  if ! command -v gcloud >/dev/null 2>&1; then
    echo "Google Cloud credentials not found at $ADC_FILE." >&2
    echo "Install gcloud or configure GOOGLE_APPLICATION_CREDENTIALS." >&2
    exit 2
  fi
  if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
    echo "Vertex AI credentials are not configured. Run: gcloud auth application-default login" >&2
    exit 2
  fi
fi

.venv/bin/pip install -r requirements.txt

exec .venv/bin/python Code/meme_persona_pipeline.py \
  --backend vertex \
  --model gemini-3.8-flash \
  --gcp-project "$GOOGLE_CLOUD_PROJECT" \
  --gcp-location "$GOOGLE_CLOUD_LOCATION" \
  --language Chinese \
  --split train \
  --limit 10 \
  --images-zip data/raw/Chinese/Train/Train_images \
  --labels-xlsx data/raw/Chinese/Train/Train_labels.xlsx \
  --persona-pool artifacts/persona_council_zh.json \
  --output artifacts/chinese_council_gemini_debate.jsonl
