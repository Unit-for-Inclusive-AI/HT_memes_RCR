"""Deterministic, evidence-based persona shortlist and debate selection."""

from __future__ import annotations

import itertools
import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SCORE_DIMENSIONS = (
    "relevance",
    "evidence_strength",
    "diversity_contribution",
    "complementarity",
)


def _scale_1_to_5(value: float) -> int:
    return max(1, min(5, 1 + round(4 * max(0.0, min(1.0, value)))))


def _comments(persona: dict[str, Any]) -> list[str]:
    values = persona.get("historical_comments", persona.get("sample_historical_comments", []))
    if not isinstance(values, list):
        return []
    # Repeated copies do not make evidence stronger. Preserve order for stable output.
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _style(persona: dict[str, Any]) -> str:
    style = persona.get("commenting_style", "")
    habits = persona.get("commenting_habits", {})
    if not style and isinstance(habits, dict):
        style = habits.get("summary", "")
    return re.sub(r"\s+", " ", str(style).strip().lower())


def _language(persona: dict[str, Any]) -> str:
    return str(persona.get("language", "Unknown")).strip().lower() or "unknown"


def _unique_ngrams(persona: dict[str, Any]) -> set[str]:
    text = " ".join(_comments(persona)).lower()
    compact = re.sub(r"\s+", "", text)
    return {compact[i : i + 3] for i in range(max(0, len(compact) - 2))}


def _pair_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    style_difference = float(_style(left) != _style(right))
    language_difference = float(_language(left) != _language(right))
    left_grams, right_grams = _unique_ngrams(left), _unique_ngrams(right)
    if not left_grams and not right_grams:
        lexical_difference = 0.0
    else:
        lexical_difference = 1 - len(left_grams & right_grams) / len(left_grams | right_grams)
    return 0.2 * language_difference + 0.3 * style_difference + 0.5 * lexical_difference


def _evidence_strength(count: int) -> int:
    # Unique source comments are the evidence quantity; no inferred stance is used.
    if count <= 1:
        return 1
    if count <= 3:
        return 2
    if count <= 5:
        return 3
    if count <= 9:
        return 4
    return 5


def select_personas(
    personas: list[dict[str, Any]],
    meme_text: str,
    shortlist_size: int = 8,
    final_count: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return scored candidates and a complementary subset, using stable rules only."""
    if shortlist_size < 1 or final_count < 1:
        raise ValueError("shortlist_size and final_count must be positive")
    if not personas:
        return [], []

    comment_sets = [_comments(persona) for persona in personas]
    documents = [" ".join(comments) for comments in comment_sets]
    try:
        vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), min_df=1, max_features=50000,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(documents + [meme_text or "meme"])
        similarities = cosine_similarity(matrix[-1], matrix[:-1]).ravel().tolist()
    except ValueError:
        similarities = [0.0] * len(personas)

    prepared: list[dict[str, Any]] = []
    for persona, comments, similarity in zip(personas, comment_sets, similarities):
        item = dict(persona)
        relevance = _scale_1_to_5(min(1.0, similarity / 0.25))
        evidence = _evidence_strength(len(comments))
        item["persona_scores"] = {
            "relevance": relevance,
            "evidence_strength": evidence,
            "diversity_contribution": 1,
            "complementarity": 1,
        }
        item["relevance_raw"] = round(float(similarity), 6)
        item["unique_comment_count"] = len(comments)
        item["selection_reason"] = ""
        prepared.append(item)

    prepared.sort(key=lambda item: item["persona_id"])
    prepared.sort(
        key=lambda item: 0.65 * item["persona_scores"]["relevance"]
        + 0.35 * item["persona_scores"]["evidence_strength"],
        reverse=True,
    )
    candidates = prepared[: min(shortlist_size, len(prepared))]

    for index, item in enumerate(candidates):
        peers = candidates[:index] + candidates[index + 1 :]
        if peers:
            diversity = sum(_pair_distance(item, peer) for peer in peers) / len(peers)
            peer_grams = set().union(*(_unique_ngrams(peer) for peer in peers))
            grams = _unique_ngrams(item)
            novelty = len(grams - peer_grams) / max(1, len(grams))
        else:
            diversity, novelty = 0.5, 0.5
        item["persona_scores"]["diversity_contribution"] = _scale_1_to_5(diversity)
        item["persona_scores"]["complementarity"] = _scale_1_to_5(novelty)

    trio_count = min(final_count, len(candidates))
    best_combo: tuple[dict[str, Any], ...] | None = None
    best_score = float("-inf")
    for combo in itertools.combinations(candidates, trio_count):
        dims = [item["persona_scores"] for item in combo]
        if len(combo) > 1:
            pairwise = [
                _pair_distance(combo[i], combo[j])
                for i in range(len(combo)) for j in range(i + 1, len(combo))
            ]
            set_diversity = sum(pairwise) / len(pairwise)
            complementarity = sum(
                len(_unique_ngrams(item) - set().union(*(
                    _unique_ngrams(other) for other in combo if other is not item
                ))) / max(1, len(_unique_ngrams(item)))
                for item in combo
            ) / len(combo)
        else:
            set_diversity = dims[0]["diversity_contribution"] / 5
            complementarity = dims[0]["complementarity"] / 5
        relevance = sum(score["relevance"] for score in dims) / (5 * len(combo))
        strength = sum(score["evidence_strength"] for score in dims) / (5 * len(combo))
        
        # Archetype diversity bonus: reward trios with distinct societal lenses
        unique_archetypes = len(set(item.get("archetype", "") for item in combo if item.get("archetype")))
        archetype_diversity = unique_archetypes / max(1, len(combo))

        score = (
            0.30 * relevance
            + 0.25 * strength
            + 0.20 * set_diversity
            + 0.15 * complementarity
            + 0.10 * archetype_diversity
        )
        tie_key = tuple(item.get("persona_id", "") for item in combo)
        current_tie = tuple(item.get("persona_id", "") for item in best_combo or ())
        if score > best_score or (score == best_score and tie_key < current_tie):
            best_combo, best_score = combo, score

    selected_ids = {item["persona_id"] for item in best_combo or ()}
    for item in candidates:
        scores = item["persona_scores"]
        if item["persona_id"] in selected_ids:
            item["selection_reason"] = (
                "Selected in the highest-scoring trio balancing meme relevance, unique-comment "
                "evidence, observable style/language diversity, and distinct comment content."
            )
        else:
            item["selection_reason"] = (
                "Shortlisted candidate; the final trio had a better combined relevance, evidence, "
                "diversity, and complementarity score."
            )
        item["selection_score"] = round(
            0.35 * scores["relevance"]
            + 0.25 * scores["evidence_strength"]
            + 0.20 * scores["diversity_contribution"]
            + 0.20 * scores["complementarity"],
            3,
        )
    selected = [item for item in candidates if item["persona_id"] in selected_ids]
    selected.sort(key=lambda item: item["persona_id"])
    return candidates, selected
