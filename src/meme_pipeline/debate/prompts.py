"""
Persona-Steered Multi-Agent Prompt Suite for Anti-LGBTQ+ Meme Classification.

Provides structured system and user prompts for:
  1. Initial Independent Persona Classification
  2. Reflect-Critique-Refine (RCR) Deliberation
  3. Impartial Critic Judge Adjudication
  4. Supportive Counter-Narrative Generation
"""

from __future__ import annotations

import json
from typing import Any

LABELS = ("Homophobic", "Transphobic", "Non_Anti_LGBT")

LABEL_DEFINITIONS = (
    "- Homophobic: Clear evidence that the meme targets, demeans, stereotypes, excludes, or expresses hostility toward gay/lesbian/bisexual people or same-sex attraction.\n"
    "- Transphobic: Clear evidence that the meme targets, demeans, stereotypes, excludes, or expresses hostility toward transgender people or gender transition.\n"
    "- Non_Anti_LGBT: Supportive, neutral, unrelated, or insufficiently evidenced content without clear anti-LGBTQ+ hostility. Includes satire/banter not directed as hate."
)


def build_initial_classification_prompt(
    persona: dict[str, Any],
    meme_summary: dict[str, Any],
) -> dict[str, str]:
    """
    Constructs the prompt for Round 1: Independent persona perspective-taking.
    The agent evaluates the meme through its assigned persona profile and writing style.
    """
    system = (
        "You are one simulated analytical persona participating in a research debate regarding online meme classification.\n"
        "Your persona is a perspective-taking simulation, not a real person. Do not produce slurs or expand hateful content unnecessarily.\n"
        "Evaluate the meme based strictly on visible evidence, context clues, and your assigned perspective.\n\n"
        "ALLOWED LABELS (Exact match required):\n"
        f"{LABEL_DEFINITIONS}\n\n"
        "GUIDELINES:\n"
        "1. Do not infer anti-LGBTQ+ hatred from LGBTQ+ imagery or terminology alone.\n"
        "2. Consider cultural sarcasm, irony, in-group humor, and internet meme tropes.\n"
        "3. Explicitly distinguish direct visual/textual evidence from subjective speculation.\n"
        "4. Respond strictly in valid JSON matching the specified schema."
    )

    history = persona.get("cleaned_comments") or persona.get(
        "sample_historical_comments", []
    )
    habits = persona.get("commenting_habits", {})

    persona_profile = {
        "persona_id": persona.get("persona_id", "Unknown"),
        "archetype": persona.get("archetype", "Neutral Observer"),
        "writing_style": persona.get("commenting_style") or habits.get("summary", ""),
        "formality": habits.get("formality", "口语化"),
        "profile_bio": persona.get("profile_bio", "社交网络普通活跃用户。"),
        "historical_comments_context": history[:4],
    }

    user = (
        f"Meme Analysis & Multimodal Cues:\n"
        f"{json.dumps(meme_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Your Assigned Persona Profile:\n"
        f"{json.dumps(persona_profile, ensure_ascii=False, indent=2)}\n\n"
        "INSTRUCTIONS:\n"
        "Adopt the perspective and tone of your assigned persona while maintaining rigorous analytical accuracy.\n"
        "Classify the meme into exactly one allowed label (Homophobic, Transphobic, or Non_Anti_LGBT).\n\n"
        "Return ONLY a JSON object with this exact schema:\n"
        "{\n"
        '  "label": "Homophobic" | "Transphobic" | "Non_Anti_LGBT",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "evidence": "<detailed textual and visual cues supporting this decision>",\n'
        '  "ambiguity": "<aspects of the meme that remain uncertain or subject to interpretation>",\n'
        '  "rationale": "<step-by-step reasoning explaining the classification from your perspective>",\n'
        f'  "persona_id": "{persona.get("persona_id", "")}"\n'
        "}"
    )

    return {"system": system, "prompt": user}


def build_rcr_refine_prompt(
    persona: dict[str, Any],
    initial_response: dict[str, Any],
    peer_responses: list[dict[str, Any]],
    meme_summary: dict[str, Any],
) -> dict[str, str]:
    """
    Constructs the prompt for Round 2 & 3: Reflect-Critique-Refine.
    The agent reflects on its own bias, critiques 2 peers, and refines its label.
    """
    system = (
        "Follow the Reflect-Critique-Refine (RCR) protocol for multimodal meme classification:\n"
        "1. REFLECT: Identify a potential cognitive bias, weakness, or overlooked nuance in your initial response.\n"
        "2. CRITIQUE: Objectively evaluate exactly two peer responses. Scrutinize their evidence, check for over-inference, "
        "and determine whether they misread satire, context, or malicious intent. Do not defer to confidence or verbosity.\n"
        "3. REFINE: Retain or adjust your classification label. You may only change your label if peers present new, "
        "verifiable textual/visual evidence. Do not conform merely for the sake of peer agreement.\n\n"
        "ALLOWED LABELS:\n"
        f"{LABEL_DEFINITIONS}\n\n"
        "Return strictly valid JSON matching the required schema."
    )

    habits = persona.get("commenting_habits", {})
    persona_brief = {
        "persona_id": persona.get("persona_id", ""),
        "archetype": persona.get("archetype", ""),
        "style": persona.get("commenting_style") or habits.get("summary", ""),
    }

    # Only provide peer responses (excluding self)
    peers_to_critique = [
        p for p in peer_responses if p.get("persona_id") != persona.get("persona_id")
    ][:2]

    user = (
        f"Meme Analysis:\n{json.dumps(meme_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Your Persona:\n{json.dumps(persona_brief, ensure_ascii=False, indent=2)}\n\n"
        f"Your Initial Response:\n{json.dumps(initial_response, ensure_ascii=False, indent=2)}\n\n"
        f"Peer Responses to Critique:\n{json.dumps(peers_to_critique, ensure_ascii=False, indent=2)}\n\n"
        "Execute the three RCR phases and return ONLY a JSON object:\n"
        "{\n"
        '  "label": "Homophobic" | "Transphobic" | "Non_Anti_LGBT",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "evidence": "<concrete evidence supporting final position>",\n'
        '  "self_critique": "<vulnerabilities or assumptions in your initial classification>",\n'
        '  "peer_critiques": [\n'
        '    {"peer_persona_id": "<id>", "assessment": "<strengths or logical fallacies in peer evidence>"}\n'
        "  ],\n"
        '  "refinement_reason": "<why you confirmed your label or why you changed it>",\n'
        '  "uncertainty": "<remaining unresolved questions>",\n'
        f'  "persona_id": "{persona.get("persona_id", "")}"\n'
        "}"
    )

    return {"system": system, "prompt": user}


def build_critic_judge_prompt(
    meme_summary: dict[str, Any],
    initial_responses: list[dict[str, Any]],
    refined_responses: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Constructs the prompt for the independent Critic Judge.
    Synthesizes the multi-agent deliberation trace and adjudicates the final dataset label.
    """
    system = (
        "You are an impartial independent Critic/Judge evaluating a multi-agent debate on anti-LGBTQ+ meme classification.\n"
        "Compare the agents' initial arguments and their RCR-refined conclusions.\n\n"
        "JUDICIAL RULES:\n"
        "1. Do NOT rely blindly on majority voting. An accurate minority grounded in verifiable text/visual evidence outweighs an ungrounded majority.\n"
        "2. Check for over-inference: Did agents invent intent that is not present in the meme?\n"
        "3. Check for cultural context: Is the meme using camp irony, reclaimed slang, or genuine derogatory hostility?\n"
        "4. If evidence of anti-LGBTQ+ hostility is weak or ambiguous, select 'Non_Anti_LGBT' and explicitly record uncertainty.\n\n"
        "ALLOWED LABELS:\n"
        f"{LABEL_DEFINITIONS}\n\n"
        "Return strictly valid JSON matching the required schema."
    )

    user = (
        f"Meme Analysis:\n{json.dumps(meme_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Initial Persona Responses:\n{json.dumps(initial_responses, ensure_ascii=False, indent=2)}\n\n"
        f"Refined Persona Responses (After RCR Debate):\n{json.dumps(refined_responses, ensure_ascii=False, indent=2)}\n\n"
        "Adjudicate the final decision and return ONLY a JSON object:\n"
        "{\n"
        '  "label": "Homophobic" | "Transphobic" | "Non_Anti_LGBT",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "evidence": "<decisive visual and textual proof deciding the verdict>",\n'
        '  "disagreement_summary": "<summary of conflicting points among debaters>",\n'
        '  "uncertainty": <float between 0.0 and 1.0>,\n'
        '  "agent_vote_counts": {"Homophobic": <int>, "Transphobic": <int>, "Non_Anti_LGBT": <int>},\n'
        '  "final_reason": "<substantive justification explaining why this label was chosen over alternatives>"\n'
        "}"
    )

    return {"system": system, "prompt": user}


def build_counter_narrative_prompt(
    meme_summary: dict[str, Any],
    judgment: dict[str, Any],
    language: str = "Chinese",
) -> dict[str, str]:
    """
    Constructs the prompt for supportive counter-speech generation when hate is detected.
    """
    system = (
        "You are a specialized counterspeech assistant creating constructive, supportive interventions for social media.\n"
        f"Target Language: {language}.\n\n"
        "GUIDELINES:\n"
        "1. Provide empathetic, respectful, and educational counterspeech that refutes harmful stereotypes.\n"
        "2. Directly address the harmful concept identified in the evidence without repeating any slurs or abusive language.\n"
        "3. Keep the counter-narrative natural, concise, and shareable on social platforms (1 to 3 sentences).\n"
        "4. Return strictly valid JSON."
    )

    user = (
        f"Meme Context:\n{json.dumps(meme_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Adjudication:\n{json.dumps(judgment, ensure_ascii=False, indent=2)}\n\n"
        "Generate a supportive counter-narrative and return ONLY a JSON object:\n"
        "{\n"
        '  "text": "<supportive, non-confrontational counterspeech in the target language>",\n'
        f'  "language": "{language}",\n'
        '  "target_harm": "<brief note on the stereotype or trope addressed>"\n'
        "}"
    )

    return {"system": system, "prompt": user}

