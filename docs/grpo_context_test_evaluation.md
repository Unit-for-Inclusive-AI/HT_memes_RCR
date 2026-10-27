# GRPO Evaluation with Test RCR Context

These commands generate held-out Chinese test traces and compare the base model
with a GRPO adapter using the same image, language, and RCR prompt format used
for training.

## 1. Generate test context traces

Set `OPENAI_API_KEY` in the environment before running this command. Processing
the full test set (`--limit 0`) makes OpenAI API calls and incurs API charges.

```bash
python Code/meme_persona_pipeline.py \
  --backend openai --model gpt-4.1-mini \
  --language Chinese --split test --limit 0 \
  --images-zip data/raw/Chinese/Test/Test_images \
  --labels-xlsx data/raw/Chinese/Test_labels.xlsx \
  --persona-pool artifacts/persona_pool_zh.json \
  --output artifacts/chinese_test_context_traces.jsonl
```

The trace generator checks the test workbook labels, creates meme analyses and
persona RCR debates, and saves the records to
`artifacts/chinese_test_context_traces.jsonl`.

## 2. Compare base and GRPO models

Replace `/path/to/your/grpo/adapter` with the directory containing the adapter
you want to evaluate.

```bash
python scripts/evaluate_grpo_context_test.py \
  --language Chinese \
  --traces artifacts/chinese_test_context_traces.jsonl \
  --adapter-path /path/to/your/grpo/adapter
```

The evaluator uses the training prompt builder and reports accuracy, macro-F1,
and classification reports for the base model and adapter. It verifies that
the traces cover the labeled test set and that trace labels match the test
workbook. The output defaults to
`artifacts/grpo_context_test_chinese.json`.

Use test results for final reporting. Use validation results for model
selection.
