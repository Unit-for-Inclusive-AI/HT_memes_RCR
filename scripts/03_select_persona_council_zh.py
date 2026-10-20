#!/usr/bin/env python3
"""
CLI Runner for Stage 1: Offline Pruning of Chinese Persona Pool.
Prunes artifacts/persona_pool_zh.json (300 candidates) -> artifacts/persona_council_zh.json (6 to 10 council members)
and writes docs/persona_council_statistics.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure src/ layout is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from meme_pipeline.personas.council_selector import PersonaCouncilSelector, DEFAULT_HT_DOMAIN_ANCHOR


def generate_markdown_report(
    council: list[dict],
    metadata: dict,
    output_path: Path,
) -> None:
    """Generates an academic-grade markdown report of the selected persona council."""
    lines = [
        "# Chinese Persona Council Statistics (Stage 1: Offline Pruning)",
        "",
        "This document details the **Core Council of Personas** selected offline from the raw 300 Chinese candidate pool (`artifacts/persona_pool_zh.json`) for the Anti-LGBTQ+ HT Meme classification pipeline.",
        "",
        "---",
        "",
        "## 1. Selection & Council Overview",
        "",
        f"- **Input Candidates Evaluated**: {metadata['total_candidates_evaluated']} valid profiles (from 300 raw personas)",
        f"- **Final Council Size**: **{metadata['council_size']} personas**",
        f"- **MMR Balance Factor (λ)**: `{metadata['lambda_mmr']}` (65% quality, 35% non-redundancy)",
        f"- **Formality Balance**: {', '.join(f'{k}: {v}' for k, v in metadata['formality_distribution'].items())}",
        "",
        "### Council Roster & Performance Metrics",
        "",
        "| Order | Persona ID | User Hash | Archetype | Formality | Substance | Relevance | Base Score | MMR Score |",
        "| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for p in council:
        lines.append(
            f"| #{p['selection_order']} | `{p['persona_id']}` | `{p['source_user_hash']}` | {p['archetype']} | "
            f"`{p.get('commenting_habits', {}).get('formality', 'N/A')}` | {p['substance_score']:.4f} | "
            f"{p['relevance_score']:.4f} | {p['base_score']:.4f} | {p['mmr_score']:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Archetype Representation",
        "",
        "The selection enforced coverage across all 6 sociological archetypes to ensure balanced debate tension during the multi-agent Reflect-Critique-Refine (RCR) rounds:",
        "",
        "| Archetype | Council Count | Role in Hate Meme Deliberation |",
        "| :--- | :---: | :--- |",
        "| **Supportive / 包容友善支持者** | "
        f"{metadata['archetype_distribution'].get('Supportive / 包容友善支持者', 0)} | Monitored minority dignity, anti-LGBTQ+ harassment, and inclusive standards |",
        "| **Critical / 批判保守者** | "
        f"{metadata['archetype_distribution'].get('Critical / 批判保守者', 0)} | Critical lens on progressive shifts, focused on social order and boundaries |",
        "| **Cultural / 传统价值观持有者** | "
        f"{metadata['archetype_distribution'].get('Cultural / 传统价值观持有者', 0)} | Grounded in classical cultural etiquette, traditional marriage, and family |",
        "| **Skeptical / 怀疑调侃者** | "
        f"{metadata['archetype_distribution'].get('Skeptical / 怀疑调侃者', 0)} | Online satirist; probes for cynicism, parody, and distinguishes jokes from malice |",
        "| **Educator / 事实科普与理性分析者** | "
        f"{metadata['archetype_distribution'].get('Educator / 事实科普与理性分析者', 0)} | Factual explainer, analyzes literal semantics, logic, and meme origin templates |",
        "| **Neutral Observer / 客观旁观者** | "
        f"{metadata['archetype_distribution'].get('Neutral Observer / 客观旁观者', 0)} | Surface-level common-sense interpretation from an everyday netizen |",
        "",
        "---",
        "",
        "## 3. Individual Persona Profiles & Historical Evidence",
        "",
    ])

    for p in council:
        habits = p.get("commenting_habits", {})
        comms = p.get("cleaned_comments", p.get("sample_historical_comments", []))
        lines.extend([
            f"### #{p['selection_order']} — Persona `{p['persona_id']}`",
            f"- **Archetype**: **{p['archetype']}**",
            f"- **User Hash**: `{p['source_user_hash']}`",
            f"- **Selection Reason**: {p['selection_reason']}",
            f"- **Scores**: Base Quality: `{p['base_score']:.4f}` (Substance: `{p['substance_score']:.4f}`, Relevance: `{p['relevance_score']:.4f}`) | MMR: `{p['mmr_score']:.4f}`",
            f"- **Writing Habits**: {habits.get('summary', 'N/A')} (Formality: `{habits.get('formality')}`, Avg chars: `{habits.get('avg_unit_length')}`, Emoji: `{habits.get('uses_emoji')}`, Punctuation: `{habits.get('emphatic_punctuation')}`)",
            "- **Historical Comments Mined from Source**:",
        ])
        for c in comms:
            lines.append(f"  - *\"{c}\"*")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 4. Next Step: Stage 2 Integration",
        "",
        "During runtime evaluation in [`Code/meme_persona_pipeline.py`](../Code/meme_persona_pipeline.py), the online meme selector will condition on the specific meme text and visual summary to dynamically select the **Top-3 complementary debaters** from this 8-member council.",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[✓] Saved documentation to {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: Offline Pruning of Chinese Persona Pool to Core Council"
    )
    parser.add_argument(
        "--input",
        default="artifacts/persona_pool_zh.json",
        help="Path to input 300 persona pool JSON",
    )
    parser.add_argument(
        "--output",
        default="artifacts/persona_council_zh.json",
        help="Path to save pruned persona council JSON",
    )
    parser.add_argument(
        "--stats-output",
        default="docs/persona_council_statistics.md",
        help="Path to save markdown statistics report",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=8,
        help="Target council size (between 6 and 10, default: 8)",
    )
    parser.add_argument(
        "--lambda-mmr",
        type=float,
        default=0.65,
        help="MMR trade-off parameter between quality and diversity (default: 0.65)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats_output)

    if not input_path.exists():
        print(f"[!] Error: Input file {input_path} does not exist.")
        sys.exit(1)

    print(f"[*] Loading raw personas from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        pool = json.load(f)

    selector = PersonaCouncilSelector(
        domain_anchor=DEFAULT_HT_DOMAIN_ANCHOR,
        lambda_mmr=args.lambda_mmr,
        council_size=args.size,
    )

    print("[*] Filtering low-information / spam profiles...")
    candidates = selector.filter_candidates(pool)
    print(f"[*] Retained {len(candidates)} qualified candidates for scoring.")

    print("[*] Computing Substance Scores (Volume + Character Length)...")
    selector.compute_substance_scores(candidates)

    print("[*] Computing HT Domain Relevance Scores (TF-IDF)...")
    selector.compute_relevance_scores(candidates)

    print(f"[*] Executing MMR Selection (Target Size: {args.size}, λ: {args.lambda_mmr}) with Archetype Guardrails...")
    council, metadata = selector.select_council(candidates)

    # Clean internal temporary keys before saving JSON
    clean_council = []
    for p in council:
        p_clean = dict(p)
        p_clean.pop("full_text", None)
        p_clean.pop("ngrams", None)
        clean_council.append(p_clean)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_council, f, ensure_ascii=False, indent=2)

    print(f"[✓] Saved {len(clean_council)} council personas to {output_path.resolve()}")

    print("[*] Generating Markdown statistics report...")
    generate_markdown_report(clean_council, metadata, stats_path)
    print("[✓] Stage 1 Offline Pruning complete!")


if __name__ == "__main__":
    main()

