#!/usr/bin/env python3
"""Run repeated persona debate -> consensus GRPO -> updated-debater evolution rounds."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True,
                        help="Training image directory or image ZIP; must contain training images only")
    parser.add_argument("--persona-pool", required=True,
                        help="Existing fixed user-profile persona pool JSON (read-only)")
    parser.add_argument("--initial-traces", default=None,
                        help="Optional existing train-only Council traces to seed round 0")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "raw"),
                        help="Dataset root containing language/Train labels for local validation")
    parser.add_argument("--language", choices=["Chinese", "English", "Hindi"], required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True,
                        help="New directory for all round traces, adapters, and summaries")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="Maximum debate/train evolution cycles")
    parser.add_argument("--rcr-rounds", type=int, default=3,
                        help="Maximum RCR rounds within each meme debate")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional cap for a small pilot; use the same value across rounds")
    parser.add_argument("--min-improvement", type=float, default=0.01,
                        help="Validation macro-F1 gain required to reset stopping patience")
    parser.add_argument("--patience", type=int, default=2,
                        help="Stop after this many consecutive rounds below min-improvement")
    args = parser.parse_args()

    if args.max_rounds < 1 or args.rcr_rounds < 1 or args.patience < 1:
        raise SystemExit("--max-rounds, --rcr-rounds, and --patience must be positive")
    image_path = Path(args.images).expanduser().resolve()
    pool_path = Path(args.persona_pool).expanduser().resolve()
    initial_traces = Path(args.initial_traces).expanduser().resolve() if args.initial_traces else None
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not image_path.exists():
        raise SystemExit(f"Training image input does not exist: {image_path}")
    if not pool_path.is_file():
        raise SystemExit(f"Persona pool does not exist: {pool_path}")
    if initial_traces is not None and not initial_traces.is_file():
        raise SystemExit(f"Initial trace file does not exist: {initial_traces}")
    if not data_root.is_dir():
        raise SystemExit(f"Data root does not exist: {data_root}")
    train_dir = (data_root / args.language / "Train").resolve()
    try:
        image_path.relative_to(train_dir)
    except ValueError as exc:
        raise SystemExit(
            f"Evolution accepts training images only; --images must be inside {train_dir}"
        ) from exc
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise SystemExit(f"Output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    evolution: dict[str, Any] = {
        "base_model": args.model,
        "language": args.language,
        "persona_pool": str(pool_path),
        "initial_traces": str(initial_traces) if initial_traces else None,
        "data_root": str(data_root),
        "image_input": str(image_path),
        "reward_target": "consensus",
        "ground_truth_labels_in_evolution_debate_prompts": False,
        "ground_truth_labels_in_grpo_reward_dataset": False,
        "rounds": [],
        "best_adapter": None,
        "stop_reason": "maximum_rounds_reached",
    }
    previous_adapter: str | None = None
    below_threshold_rounds = 0

    for round_index in range(args.max_rounds):
        round_dir = output_dir / f"round_{round_index:02d}"
        debate_dir = round_dir / "debate"
        train_dir = round_dir / "training"
        debate_dir.mkdir(parents=True)
        train_dir.mkdir(parents=True)
        traces_path = debate_dir / "debate_traces.jsonl"

        use_seed_traces = round_index == 0 and initial_traces is not None
        if use_seed_traces:
            with initial_traces.open(encoding="utf-8") as stream:
                seed_rows = [json.loads(line) for line in stream if line.strip()]
            seed_ids = []
            for row in seed_rows:
                if row.get("split") != "train" or row.get("language") != args.language:
                    raise SystemExit(
                        "--initial-traces must contain only records with split='train' "
                        f"and language={args.language!r}."
                    )
                seed_ids.append(row.get("image_id"))
            if not seed_rows or len(seed_ids) != len(set(seed_ids)):
                raise SystemExit("--initial-traces must be nonempty and have unique image IDs.")
            shutil.copy2(initial_traces, traces_path)
            print(
                f"Using {len(seed_rows)} existing Council traces to seed evolution round 0: "
                f"{initial_traces}"
            )
        else:
            debate_command = [
                sys.executable, str(PROJECT_ROOT / "Code" / "meme_persona_pipeline.py"),
                "--backend", "local",
                "--model", args.model,
                "--images-zip", str(image_path),
                "--image-dir", str(debate_dir / "prepared_images"),
                "--persona-pool", str(pool_path),
                "--language", args.language,
                "--split", "train",
                "--limit", str(args.max_samples) if args.max_samples is not None else "0",
                "--output", str(traces_path),
                "--hide-labels",
                "--rcr-rounds", str(args.rcr_rounds),
                "--evolution-round", str(round_index),
                "--consensus-only",
                "--skip-counter-narrative",
            ]
            if previous_adapter:
                debate_command.extend(["--adapter-path", previous_adapter])
            run(debate_command)

        train_command = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "train_grpo_vlm.py"),
            "--traces", str(traces_path),
            "--data-root", str(data_root),
            "--output-dir", str(train_dir),
            "--model", args.model,
            "--reward-target", "consensus",
            "--epochs", str(args.epochs),
            "--learning-rate", str(args.learning_rate),
            "--num-generations", str(args.num_generations),
            "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
            "--validation-fraction", str(args.validation_fraction),
            "--seed", str(args.seed),
        ]
        if args.max_samples is not None:
            train_command.extend(["--max-samples", str(args.max_samples)])
        if previous_adapter:
            train_command.extend(["--init-adapter", previous_adapter])
        run(train_command)

        summary_path = train_dir / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        base_f1 = summary["base_validation"]["macro_f1"]
        trained_f1 = summary["trained_validation"]["macro_f1"]
        gain = trained_f1 - base_f1
        promoted = bool(summary.get("promoted"))
        adapter_path = str((train_dir / "final_adapter").resolve())
        round_record = {
            "round": round_index,
            "source_adapter": previous_adapter,
            "adapter": adapter_path,
            "trace_file": str(traces_path.resolve()),
            "base_validation_macro_f1": base_f1,
            "trained_validation_macro_f1": trained_f1,
            "validation_macro_f1_gain": gain,
            "promoted": promoted,
        }
        evolution["rounds"].append(round_record)

        if not promoted:
            evolution["stop_reason"] = "round_not_promoted"
            break

        previous_adapter = adapter_path
        evolution["best_adapter"] = adapter_path
        if gain >= args.min_improvement:
            below_threshold_rounds = 0
        else:
            below_threshold_rounds += 1
        if below_threshold_rounds >= args.patience:
            evolution["stop_reason"] = "validation_improvement_plateau"
            break

    summary_path = output_dir / "evolution_summary.json"
    summary_path.write_text(json.dumps(evolution, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(evolution, ensure_ascii=False, indent=2))
    print(f"Evolution summary saved to {summary_path}")


if __name__ == "__main__":
    main()
