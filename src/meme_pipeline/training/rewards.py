"""Small dependency-free reward helpers, also used by unit tests."""

from __future__ import annotations

import json
import re
from typing import Any

LABELS = {"Homophobic", "Transphobic", "Non_Anti_LGBT"}


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("role") == "assistant":
                    content = item.get("content", "")
                    if isinstance(content, list):
                        parts.extend(
                            str(part.get("text", "")) for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        )
                    else:
                        parts.append(str(content))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", completion.get("text", "")))
    return str(completion or "")


def parse_completion(completion: Any) -> dict[str, Any] | None:
    text = completion_text(completion).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        try:
            result = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            result = None
    return result if isinstance(result, dict) else None


def classification_accuracy_reward(completions, gold_label, **kwargs) -> list[float]:
    del kwargs
    labels = gold_label if isinstance(gold_label, list) else [gold_label] * len(completions)
    return [
        1.0 if (parsed := parse_completion(completion)) is not None
        and parsed.get("label") == expected else 0.0
        for completion, expected in zip(completions, labels)
    ]


def consensus_accuracy_reward(completions, consensus_label, **kwargs) -> list[float]:
    """Reward agreement with the debate consensus, without using dataset gold labels."""
    del kwargs
    labels = (
        consensus_label
        if isinstance(consensus_label, list)
        else [consensus_label] * len(completions)
    )
    return [
        1.0 if (parsed := parse_completion(completion)) is not None
        and parsed.get("label") == expected else 0.0
        for completion, expected in zip(completions, labels)
    ]


def structured_output_reward(completions, **kwargs) -> list[float]:
    del kwargs
    return [
        0.1 if (parsed := parse_completion(completion)) is not None
        and parsed.get("label") in LABELS
        and "confidence" in parsed
        and "evidence" in parsed
        else 0.0
        for completion in completions
    ]


def evidence_reward(completions, **kwargs) -> list[float]:
    del kwargs
    scores = []
    for completion in completions:
        parsed = parse_completion(completion)
        evidence = parsed.get("evidence") if parsed else None
        if isinstance(evidence, str):
            nonempty = bool(evidence.strip())
        elif isinstance(evidence, (list, dict)):
            nonempty = bool(evidence)
        else:
            nonempty = False
        scores.append(0.1 if nonempty else 0.0)
    return scores
