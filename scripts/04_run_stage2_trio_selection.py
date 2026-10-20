#!/usr/bin/env python3
"""
Stage 2: Dynamic Top-3 Persona Selection & Prompt Construction.

Given an input meme:
  1. Scores council personas on a 1-5 scale (Relevance, Strength, Diversity, Complementarity).
  2. Selects the optimal Top-3 complementary personas (enforcing distinct archetypes).
  3. Builds production prompts for LLM agents (Initial, RCR Refine, Critic Judge, Counter-Narrative).
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

from meme_pipeline.personas.selector import select_personas
from meme_pipeline.debate.prompts import (
    build_initial_classification_prompt,
    build_rcr_refine_prompt,
    build_critic_judge_prompt,
    build_counter_narrative_prompt,
)


def get_mock_meme_analysis(image_path: Path) -> dict:
    """Fallback / mock meme analysis for dry-run verification."""
    stem = image_path.stem
    return {
        "image_file": image_path.name,
        "visible_text": "跨沙小，沒看過gay喔！",
        "visual_description": "画面使用动漫角色挑衅自信的邪魅表情，搭配粗口谐音台词，呈现网络常见的拽酷搞怪风格。",
        "context_clues": "源自二次元亚文化与网络日常自嘲表情包，需判断是无恶意调侃还是反LGBTQ+敌意贬损。",
        "uncertainty": "台词语气带有攻击性口吻，但语境更偏向动漫圈内部的傲娇自信反应图。",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Dynamic Top-3 Persona Selection and RCR Prompt Generation"
    )
    parser.add_argument(
        "--image",
        default="data/raw/Chinese/Train/Train_images/1.jpg",
        help="Path to target meme image",
    )
    parser.add_argument(
        "--council",
        default="artifacts/persona_council_zh.json",
        help="Path to curated persona council JSON",
    )
    parser.add_argument(
        "--output-prompts",
        default="artifacts/stage2_generated_prompts.json",
        help="Path to save generated prompts JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run without calling external vision APIs (default: True for cost safety)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    council_path = Path(args.council)

    if not council_path.exists():
        print(f"[!] Error: Council file not found at {council_path}")
        sys.exit(1)

    print(f"[*] Loading Persona Council from {council_path}...")
    with open(council_path, "r", encoding="utf-8") as f:
        council = json.load(f)
    print(f"[✓] Loaded {len(council)} Council members.")

    print(f"[*] Analyzing target meme: {image_path}...")
    meme_summary = get_mock_meme_analysis(image_path)
    ranking_text = f"{meme_summary.get('visible_text', '')} {meme_summary.get('visual_description', '')} {meme_summary.get('context_clues', '')}"

    print("[*] Judging & scoring council members against meme content on 1-5 scale...")
    candidates, selected = select_personas(
        council, ranking_text, shortlist_size=len(council), final_count=3
    )

    print("\n" + "=" * 80)
    print("STAGE 2: 1–5 JUDGING SCORECARD & TRIO SELECTION RESULTS")
    print("=" * 80)
    print(f"Meme Image: {image_path.name}")
    print(f"Visible Text: {meme_summary.get('visible_text')}")
    print("-" * 80)
    print(f"{'Persona ID':<12} | {'Archetype':<26} | {'Rel':<4} | {'Str':<4} | {'Div':<4} | {'Comp':<5} | {'Decision'}")
    print("-" * 80)

    selected_ids = {p["persona_id"] for p in selected}
    for p in candidates:
        scores = p.get("persona_scores", {})
        pid = p.get("persona_id")
        arch = p.get("archetype", "")[:24]
        rel = scores.get("relevance", 1)
        st = scores.get("evidence_strength", 1)
        div = scores.get("diversity_contribution", 1)
        comp = scores.get("complementarity", 1)
        status = "★ SELECTED DEBATER" if pid in selected_ids else "  Shortlisted"
        print(f"{pid:<12} | {arch:<26} | {rel:<4} | {st:<4} | {div:<4} | {comp:<5} | {status}")

    print("=" * 80)
    print(f"\n[✓] Successfully selected {len(selected)} debaters for RCR multi-agent debate:")
    for idx, debater in enumerate(selected, 1):
        print(f"  Debater {idx}: {debater['persona_id']} ({debater['archetype']}) — Style: {debater.get('commenting_habits', {}).get('formality')}")

    print("\n[*] Constructing multi-agent debate prompts...")
    prompts_payload = {
        "meme_analysis": meme_summary,
        "selected_debaters": [
            {
                "persona_id": d["persona_id"],
                "archetype": d["archetype"],
                "scores": d["persona_scores"],
            }
            for d in selected
        ],
        "initial_prompts": {},
        "rcr_refine_sample": {},
        "critic_judge_prompt": {},
        "counter_narrative_prompt": {},
    }

    # 1. Initial Prompts
    for d in selected:
        pid = d["persona_id"]
        init_p = build_initial_classification_prompt(d, meme_summary)
        prompts_payload["initial_prompts"][pid] = init_p

    # 2. RCR Refine Prompt (Sample for Debater 1 critiquing Debater 2 & 3)
    mock_initial_resp = {
        "label": "Non_Anti_LGBT",
        "confidence": 0.85,
        "evidence": "画面使用动漫搞笑表情自嘲，台语俚语属于夸张情绪表达。",
        "ambiguity": "用语包含gay字眼但无恶意倾向。",
        "rationale": "二次元动漫受众常见的reaction meme，缺乏歧视事实。",
        "persona_id": selected[0]["persona_id"],
    }
    mock_peers = [
        {
            "persona_id": selected[1]["persona_id"],
            "label": "Homophobic",
            "confidence": 0.70,
            "evidence": "文本使用带有冒犯性的反问句。",
            "ambiguity": "语气可能过于粗鲁。",
            "rationale": "在公共空间使用带有攻击色彩的称谓可能强化偏见。",
        },
        {
            "persona_id": selected[2]["persona_id"],
            "label": "Non_Anti_LGBT",
            "confidence": 0.80,
            "evidence": "动漫角色DIO经典表情包衍生。",
            "ambiguity": "需结合网络流行语理解。",
            "rationale": "属于圈层内部玩梗，并未贬低群体价值。",
        },
    ]
    refine_p = build_rcr_refine_prompt(
        selected[0], mock_initial_resp, mock_peers, meme_summary
    )
    prompts_payload["rcr_refine_sample"] = refine_p

    # 3. Critic Judge Prompt
    judge_p = build_critic_judge_prompt(
        meme_summary,
        [mock_initial_resp, mock_peers[0], mock_peers[1]],
        [mock_initial_resp, mock_peers[1]],
    )
    prompts_payload["critic_judge_prompt"] = judge_p

    # 4. Counter-Narrative Prompt
    mock_verdict = {
        "label": "Homophobic",
        "confidence": 0.82,
        "evidence": "文本刻板调侃性少数群体。",
        "agent_vote_counts": {"Homophobic": 2, "Non_Anti_LGBT": 1, "Transphobic": 0},
        "final_reason": "借用网络玩梗外衣传播贬损性概念。",
    }
    cn_p = build_counter_narrative_prompt(meme_summary, mock_verdict, language="Chinese")
    prompts_payload["counter_narrative_prompt"] = cn_p

    # Save prompts artifact
    out_path = Path(args.output_prompts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prompts_payload, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] Saved complete prompt suite to: {out_path.resolve()}")
    print("\n" + "=" * 80)
    print("SAMPLE GENERATED PROMPT (AGENT 1 INITIAL PERSPECTIVE PROMPT)")
    print("=" * 80)
    agent_1_prompt = prompts_payload["initial_prompts"][selected[0]["persona_id"]]
    print(f"--- SYSTEM PROMPT ---\n{agent_1_prompt['system']}\n")
    print(f"--- USER PROMPT ---\n{agent_1_prompt['prompt'][:500]}...\n[Truncated for display]")
    print("=" * 80)


if __name__ == "__main__":
    main()

