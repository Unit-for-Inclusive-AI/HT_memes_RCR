#!/usr/bin/env python3
"""Train a Qwen2.5-VL classifier from train-only RCR debate traces with GRPO."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meme_pipeline.training.data import load_training_rows, read_trace_files, split_train_validation
from meme_pipeline.training.rewards import (
    classification_accuracy_reward,
    evidence_reward,
    structured_output_reward,
    parse_completion,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LABELS = ["Homophobic", "Transphobic", "Non_Anti_LGBT"]


def image_data_url(path: str) -> str:
    from PIL import Image
    import io

    with Image.open(path) as source:
        image = source.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def predict_rows(model_name: str, adapter_path: str | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from meme_pipeline.models.local_vlm import LocalVisionLanguageClient
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    client = LocalVisionLanguageClient.from_pretrained(model_name, adapter_path)
    gold, predictions = [], []
    for row in rows:
        prompt_messages = row["prompt"]
        system = next((m["content"] for m in prompt_messages if m["role"] == "system"), "")
        user_text = next((m["content"] for m in prompt_messages if m["role"] == "user"), "")
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data_url(row["image_path"])}}
                ]},
            ],
            max_tokens=512,
        )
        parsed = parse_completion(response.choices[0].message.content)
        label = parsed.get("label") if parsed else None
        predictions.append(label if label in LABELS else "__invalid__")
        gold.append(row["gold_label"])
    per_language = {}
    for language in sorted({row["language"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["language"] == language]
        per_language[language] = {
            "n": len(indices),
            "accuracy": accuracy_score([gold[i] for i in indices], [predictions[i] for i in indices]),
            "macro_f1": f1_score(
                [gold[i] for i in indices], [predictions[i] for i in indices],
                labels=LABELS, average="macro", zero_division=0,
            ),
        }
    return {
        "n": len(rows),
        "accuracy": accuracy_score(gold, predictions),
        "macro_f1": f1_score(gold, predictions, labels=LABELS, average="macro", zero_division=0),
        "classification_report": classification_report(
            gold, predictions, labels=LABELS, output_dict=True, zero_division=0
        ),
        "per_language": per_language,
    }


def materialize_dataset(rows: list[dict[str, Any]]):
    from datasets import Dataset
    from PIL import Image

    examples = []
    for row in rows:
        with Image.open(row["image_path"]) as image:
            examples.append({
                "image": image.convert("RGB").copy(),
                "prompt": row["prompt"],
                "gold_label": row["gold_label"],
                "language": row["language"],
            })
    return Dataset.from_list(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True,
                        help="JSONL debate trace files generated with --split train")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts" / "grpo_meme_model"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional small subset for a first GPU smoke run")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Completions sampled per prompt; use a small value on limited VRAM")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=256)
    args = parser.parse_args()

    if args.num_generations < 2:
        raise SystemExit("--num-generations must be at least 2 for a GRPO comparison group.")
    if args.gradient_accumulation_steps % args.num_generations:
        raise SystemExit(
            "For this single-GPU setup, gradient_accumulation_steps must be divisible by "
            "num_generations. The defaults (4 and 4) are suitable for a Colab T4 pilot."
        )

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoProcessor
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "GRPO training dependencies are missing. On a GPU host, install "
            "requirements-training.txt first."
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("GRPO VLM training requires a CUDA GPU; none is available in this environment.")

    records = read_trace_files(args.traces)
    rows = load_training_rows(records, args.data_root, max_samples=args.max_samples)
    train_rows, validation_rows = split_train_validation(
        rows, validation_fraction=args.validation_fraction, seed=args.seed
    )
    if len(train_rows) < args.num_generations:
        raise SystemExit(
            f"GRPO needs at least {args.num_generations} training examples for the configured group size."
        )
    if not validation_rows:
        raise SystemExit("Validation split is empty; provide more training traces.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model)

    print(f"Evaluating base model on {len(validation_rows)} validation memes...")
    base_metrics = predict_rows(args.model, None, validation_rows)
    (output_dir / "base_validation_metrics.json").write_text(
        json.dumps(base_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_dataset = materialize_dataset(train_rows)
    processor = AutoProcessor.from_pretrained(args.model)
    use_bf16 = torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    config = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        model_init_kwargs={"torch_dtype": model_dtype},
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[classification_accuracy_reward, structured_output_reward, evidence_reward],
        args=config,
        train_dataset=train_dataset,
        processing_class=processor,
        peft_config=LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        ),
    )
    trainer.train()
    adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(output_dir / "processor")
    del trainer, train_dataset, processor
    torch.cuda.empty_cache()

    print(f"Evaluating trained adapter on {len(validation_rows)} validation memes...")
    tuned_metrics = predict_rows(args.model, str(adapter_dir), validation_rows)
    validation_labels = {row["gold_label"] for row in validation_rows}
    promotion_eligible = len(validation_rows) >= 30 and set(LABELS) <= validation_labels
    promoted = promotion_eligible and tuned_metrics["macro_f1"] > base_metrics["macro_f1"]
    summary = {
        "base_model": args.model,
        "adapter_path": str(adapter_dir),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "seed": args.seed,
        "grpo_group_size": args.num_generations,
        "promotion_eligible": promotion_eligible,
        "rewards": {
            "classification_accuracy": 1.0,
            "valid_structured_output": 0.1,
            "nonempty_evidence": 0.1,
        },
        "base_validation": base_metrics,
        "trained_validation": tuned_metrics,
        "promoted": promoted,
        "inference_adapter_path": str(adapter_dir) if promoted else None,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "training_config.json").write_text(json.dumps({
        "model": args.model,
        "data_root": os.path.abspath(args.data_root),
        "traces": [os.path.abspath(path) for path in args.traces],
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "num_generations": args.num_generations,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "peft": {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"]},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "base_macro_f1": base_metrics["macro_f1"],
        "trained_macro_f1": tuned_metrics["macro_f1"],
        "promoted": promoted,
        "adapter": str(adapter_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
