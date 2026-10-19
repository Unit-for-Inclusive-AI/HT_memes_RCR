"""Counter-narrative gating and prompt construction."""

from __future__ import annotations

import json
import re
from typing import Any, Callable


MIN_COUNTER_NARRATIVE_CONFIDENCE = 0.70


def has_unresolved_uncertainty(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value > 0
    text = str(value).strip().lower()
    if not text or text in {"no", "none", "false", "low", "not applicable", "n/a"}:
        return False
    if text.startswith(("no ", "none ", "not ", "false ")):
        return False
    if text in {"yes", "true", "high", "medium", "uncertain", "ambiguous"}:
        return True
    return True


def counter_narrative_status(
    classification: dict[str, Any],
    confidence_threshold: float = MIN_COUNTER_NARRATIVE_CONFIDENCE,
) -> str:
    label = classification.get("label")
    if label == "Non_Anti_LGBT":
        return "not_applicable"
    if label not in {"Homophobic", "Transphobic"}:
        return "needs_review"
    try:
        confidence = float(classification.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < confidence_threshold or has_unresolved_uncertainty(classification.get("uncertainty")):
        return "needs_review"
    return "generated"


def build_counter_narrative_prompt(
    language: str,
    classification: dict[str, Any],
    meme_analysis: dict[str, Any],
) -> str:
    evidence = classification.get("evidence", [])
    if not isinstance(evidence, (str, list, dict)):
        evidence = str(evidence)
    context = {
        "label": classification.get("label"),
        "final_reason": classification.get("final_reason", classification.get("rationale", "")),
        "evidence": evidence,
        "disagreement_summary": classification.get("disagreement_summary", ""),
        "meme_analysis": meme_analysis,
    }
    return (
        "Write one short, shareable counter-narrative in the requested language. Respond to the "
        "specific harmful idea supported by the evidence, affirm dignity and inclusion, and avoid "
        "repeating slurs or reproducing the harmful claim. Do not invent context, attribute intent "
        "beyond the evidence, or infer anyone's identity from appearance. Keep it to one or two "
        "sentences. Return JSON only with keys text and language.\n\n"
        f"Requested language: {language}\n"
        f"Classification evidence and reasoning (use as grounding, not as text to quote):\n"
        f"{json.dumps(context, ensure_ascii=False)}"
    )


def generate_counter_narrative(
    classification: dict[str, Any],
    meme_analysis: dict[str, Any],
    language: str,
    generator: Callable[[str], str],
    confidence_threshold: float = MIN_COUNTER_NARRATIVE_CONFIDENCE,
) -> dict[str, Any]:
    status = counter_narrative_status(classification, confidence_threshold)
    if status != "generated":
        return {"status": status, "text": None, "language": language}
    raw = generator(build_counter_narrative_prompt(language, classification, meme_analysis))
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        try:
            parsed = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    text = parsed.get("text")
    if not isinstance(text, str) or not text.strip():
        return {
            "status": "generation_failed",
            "text": None,
            "language": language,
            "raw_response": raw,
        }
    return {"status": "generated", "text": text.strip(), "language": language}
