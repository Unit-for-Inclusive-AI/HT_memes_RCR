# Persona Debate and GRPO Evolution

`scripts/run_grpo_evolution.py` runs repeated local-model cycles:

1. The current Qwen2.5-VL model (base model in round 0, last promoted adapter afterward) analyzes each training meme, selects personas from the existing fixed pool with the existing selection procedure, and runs up to three RCR rounds.
2. The three refined persona labels produce a majority-vote `consensus_label`.
3. GRPO trains on those traces using consensus agreement, JSON structure, and nonempty evidence rewards. Ground-truth labels are not loaded during debate generation and are omitted from the GRPO training dataset in consensus mode.
4. The candidate adapter is evaluated against a deterministic validation split from the training set. The next evolution round uses the candidate only when the current training script promotes it. The loop stops on a rejected candidate, validation plateau, or the configured maximum number of cycles.

The existing persona-pool file and previous experiment artifacts are read-only inputs. Each run writes traces, checkpoints, adapters, and summaries under a new output directory. Supply a training image folder or ZIP only; the training loader checks every trace ID against the selected language's `Train_labels.xlsx` and `Train` images. Test data is not accepted as a training source.

## Chinese example

Run this on the CUDA training environment. Use an output directory that does not already contain files.

```bash
python scripts/run_grpo_evolution.py \
  --language Chinese \
  --images data/raw/Chinese/Train/Train_images \
  --persona-pool artifacts/persona_pool_zh.json \
  --data-root data/raw \
  --output-dir artifacts/chinese_grpo_evolution_v1 \
  --max-rounds 3 \
  --rcr-rounds 3 \
  --epochs 1
```

For an initial wiring pilot, add `--max-samples 32`; this runs debate and one training attempt but will not pass the current 30-example validation promotion gate, so it will not continue to a later evolution round. The default output is not a full paper reproduction: the model is multimodal, personas are selected from your user-profile pool, and the training reward uses the Council's consensus label rather than the paper's complete reward design.

## Outputs

Each `round_NN/` directory contains `debate/debate_traces.jsonl` and `training/`, including the candidate adapter, validation metrics, and training configuration. `evolution_summary.json` records each round's source and resulting adapter, validation macro-F1 change, promotion result, and stopping reason. The first round starts from the base model; later rounds start from the previously promoted adapter.

To use a gold-label GRPO run instead, continue to call `scripts/train_grpo_vlm.py` directly with its default `--reward-target gold`. The evolution orchestrator intentionally sets `--reward-target consensus`.
