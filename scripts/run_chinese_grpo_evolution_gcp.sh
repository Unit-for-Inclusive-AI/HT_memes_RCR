#!/usr/bin/env bash
set -euo pipefail

# Run from a GCP GPU VM (including VS Code Remote SSH). Data can live on the VM
# or on a mounted Cloud Storage/FUSE path.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/raw}"
IMAGES="${IMAGES:-${DATA_ROOT}/Chinese/Train/Train_images}"
PERSONA_POOL="${PERSONA_POOL:-${PROJECT_ROOT}/artifacts/persona_pool_zh.json}"
INITIAL_TRACES="${INITIAL_TRACES:-${PROJECT_ROOT}/artifacts/chinese_council_gpt_image_specific_300_v3_debate.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
RCR_ROUNDS="${RCR_ROUNDS:-3}"
EPOCHS="${EPOCHS:-1}"

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "Set OUTPUT_DIR to a new or empty directory before running." >&2
  echo 'Example: export OUTPUT_DIR="/mnt/data/chinese_grpo_evolution_run_01"' >&2
  exit 2
fi

for input_path in "${IMAGES}" "${PERSONA_POOL}" "${INITIAL_TRACES}" "${DATA_ROOT}/Chinese/Train/Train_labels.xlsx"; do
  if [[ ! -e "${input_path}" ]]; then
    echo "Required training input not found: ${input_path}" >&2
    exit 2
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; use a CUDA-enabled GPU VM." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA GPU is not available"'
nvidia-smi

args=(
  --language Chinese
  --images "${IMAGES}"
  --persona-pool "${PERSONA_POOL}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --max-rounds "${MAX_ROUNDS}"
  --rcr-rounds "${RCR_ROUNDS}"
  --initial-traces "${INITIAL_TRACES}"
  --epochs "${EPOCHS}"
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  args+=(--max-samples "${MAX_SAMPLES}")
fi

exec python scripts/run_grpo_evolution.py "${args[@]}"
