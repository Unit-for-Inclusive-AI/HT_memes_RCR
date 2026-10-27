# Persona DTE meme workflow

The classifier shortlists eight personas, scores relevance, evidence strength,
diversity, and complementarity, then runs RCR with the best complementary trio.
Each per-meme JSONL record includes the shortlist, chosen personas, RCR trace,
judge result, and the counter-narrative status or generated text.

## Generate train traces

From the repository root, generate a separate trace file for each language. The
input image directory and label workbook must belong to the same training split.

```bash
python Code/meme_persona_pipeline.py \
  --language Chinese --split train \
  --images-zip data/raw/Chinese/Train/Train_images \
  --labels-xlsx data/raw/Chinese/Train/Train_labels.xlsx \
  --output artifacts/dte_traces_zh.jsonl --limit 0

python Code/meme_persona_pipeline.py \
  --language English --split train \
  --images-zip data/raw/English/Train/Train_images \
  --labels-xlsx data/raw/English/Train/Train_labels.xlsx \
  --output artifacts/dte_traces_en.jsonl --limit 0

python Code/meme_persona_pipeline.py \
  --language Hindi --split train \
  --images-zip data/raw/Hindi/Train/Train_Images \
  --labels-xlsx data/raw/Hindi/Train/Train_labels.xlsx \
  --output artifacts/dte_traces_hi.jsonl --limit 0
```

For the Koo-derived Hindi persona pool, use the prepared five-image pilot runner:

```bash
bash scripts/run_hindi_gpt_pilot.sh
```

This uses `artifacts/persona_pool_hi_koo_provisional_300.json` and writes
`artifacts/hindi_koo_persona_gpt_pilot_5_debate.jsonl`. It makes OpenAI API
calls and uses only five Hindi training images. Review the resulting traces
before expanding the run. The pool contains 300 balanced, pseudonymous Koo
personas, but its language labels are automatic and have not been checked by
a Hindi-speaking reviewer; see
[`docs/Hindi_persona_pool_statistics.md`](docs/Hindi_persona_pool_statistics.md).

The default backend uses the OpenAI API to create meme descriptions and RCR
traces. Gold labels are stored after inference for training rewards and reporting;
they are not included in prompts. A local model can instead be used with
`--backend local --model Qwen/Qwen2.5-VL-3B-Instruct`; add
`--adapter-path artifacts/grpo_meme_model/final_adapter` after training.

## Install the GRPO training stack

Run this on a CUDA GPU host. This workspace does not include a CUDA GPU or the
training dependencies.

```bash
python -m pip install -r requirements.txt -r requirements-training.txt
```

## Train and evaluate on validation

```bash
python scripts/train_grpo_vlm.py \
  --traces artifacts/dte_traces_zh.jsonl artifacts/dte_traces_en.jsonl artifacts/dte_traces_hi.jsonl \
  --data-root data/raw \
  --output-dir artifacts/grpo_meme_model
```

The trainer rejects traces marked `split: test`, checks each recorded label
against that language's training workbook, and creates a per-language
train/validation split. It samples eight completions per prompt and rewards
correct labels (1.0), valid structured output (0.1), and non-empty evidence
(0.1). Base and adapter macro-F1 are compared on validation; the summary marks
the adapter promoted only when validation macro-F1 improves.

## Evaluate with the trained model and generate counter-narratives

Only a promoted adapter should be used for final evaluation. Point the existing
pipeline at one language's test images and labels; the test labels are used only
for the final report after prediction.

```bash
python Code/meme_persona_pipeline.py \
  --backend local --model Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter-path artifacts/grpo_meme_model/final_adapter \
  --language Chinese --split test \
  --images-zip data/raw/Chinese/Test/Test_images \
  --labels-xlsx data/raw/Chinese/Test_labels.xlsx \
  --output artifacts/test_predictions_zh.jsonl --limit 0
```

For confident `Homophobic` or `Transphobic` predictions, the last stage asks the
configured model for a short supportive counter-narrative in the meme's language.
`Non_Anti_LGBT` results are marked `not_applicable`; confidence below 0.70 or
unresolved uncertainty is marked `needs_review` with no generated text.
