"""
Modular Persona Council Selector (Stage 1: Offline Pruning).

Prunes a raw persona pool (e.g., 300 candidates) down to a high-signal
'Core Council' of 6 to 10 distinct, substantive personas for Chinese HT meme classification.
Uses:
  1. Substance & Quality Scoring (comment count, character length, deduplication).
  2. Domain Relevance Scoring (TF-IDF against HT meme discourse lexicons).
  3. Maximal Marginal Relevance (MMR) with character 3-gram Jaccard similarity.
  4. Archetype & Stylometric Coverage Guardrails.
"""

from __future__ import annotations

import re
from typing import Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Core HT Domain Anchor Lexicon for Chinese Meme Classification
DEFAULT_HT_DOMAIN_ANCHOR = (
    "传统 家庭 婚姻 道德 礼序 伦理 "
    "性别 平权 少数群体 尊重 歧视 取向 同志 "
    "调侃 讽刺 批判 客观 观点 恶意 恶搞"
)

CORE_ARCHETYPES = [
    "Supportive / 包容友善支持者",
    "Critical / 批判保守者",
    "Cultural / 传统价值观持有者",
    "Skeptical / 怀疑调侃者",
    "Educator / 事实科普与理性分析者",
    "Neutral Observer / 客观旁观者",
]


def clean_comment(text: str) -> str:
    """Strips URLs, user @mentions, and redundant whitespace from social comments."""
    t = re.sub(r"https?://\S+", "", str(text))
    t = re.sub(r"@[a-zA-Z0-9_\u4e00-\u9fff\-]+", "", t)
    return re.sub(r"\s+", " ", t).strip()


def char_3grams(text: str) -> set[str]:
    """Extracts compact character-level 3-grams for non-redundancy / Jaccard calculation."""
    compact = re.sub(r"\s+", "", str(text)).lower()
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[i : i + 3] for i in range(len(compact) - 2)}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Computes Jaccard similarity between two character n-gram sets."""
    if not set_a and not set_b:
        return 0.0
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


class PersonaCouncilSelector:
    """Selects a balanced, substantive council from a large candidate persona pool."""

    def __init__(
        self,
        domain_anchor: str = DEFAULT_HT_DOMAIN_ANCHOR,
        lambda_mmr: float = 0.65,
        council_size: int = 8,
        min_unique_comments: int = 2,
    ) -> None:
        self.domain_anchor = domain_anchor
        self.lambda_mmr = lambda_mmr
        self.council_size = max(6, min(10, council_size))
        self.min_unique_comments = min_unique_comments

    def filter_candidates(self, pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filters out low-information, spammy, or empty profiles."""
        candidates = []
        for p in pool:
            raw_comms = p.get("sample_historical_comments", [])
            cleaned = [clean_comment(c) for c in raw_comms]
            # Exclude trivial non-textual fragments
            cleaned = [c for c in cleaned if len(c) >= 3]
            unique_comms = list(dict.fromkeys(cleaned))

            if len(unique_comms) < self.min_unique_comments:
                continue

            # Ensure there is actual discourse content
            full_text = " ".join(unique_comms)
            if len(full_text.strip()) < 10:
                continue

            p_copy = dict(p)
            p_copy["cleaned_comments"] = unique_comms
            p_copy["full_text"] = full_text
            p_copy["ngrams"] = char_3grams(full_text)
            candidates.append(p_copy)
        return candidates

    def compute_substance_scores(self, candidates: list[dict[str, Any]]) -> None:
        """Computes Substance_Score(P) in [0, 1] based on volume, length, and content richness."""
        for p in candidates:
            comms = p["cleaned_comments"]
            count = len(comms)
            tot_chars = sum(len(c) for c in comms)
            avg_len = tot_chars / max(count, 1)

            # Volume score (4 comments = 1.0, 3 = 0.75, 2 = 0.50)
            vol_score = min(1.0, count / 4.0)

            # Length score (target avg 25+ characters for 1.0)
            len_score = min(1.0, avg_len / 25.0)

            # Composite substance score
            p["substance_score"] = round(0.40 * vol_score + 0.60 * len_score, 4)

    def compute_relevance_scores(self, candidates: list[dict[str, Any]]) -> None:
        """Computes Relevance_Score(P) in [0, 1] via TF-IDF against HT domain lexicon."""
        docs = [p["full_text"] for p in candidates]
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(docs + [self.domain_anchor])
        sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()

        max_sim = float(np.max(sims)) if len(sims) > 0 and np.max(sims) > 0 else 1.0
        min_sim = float(np.min(sims)) if len(sims) > 0 else 0.0

        for i, p in enumerate(candidates):
            raw_s = float(sims[i])
            if max_sim > min_sim:
                norm_s = (raw_s - min_sim) / (max_sim - min_sim)
            else:
                norm_s = 0.5
            p["relevance_score"] = round(norm_s, 4)

            # Base Quality Score = 0.50 * Substance + 0.50 * Relevance
            p["base_score"] = round(
                0.50 * p["substance_score"] + 0.50 * p["relevance_score"], 4
            )

    def select_council(
        self, candidates: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Executes MMR selection with Archetype & Stylometric Guardrails.
        Guarantees coverage across the core archetypes before allowing duplicates.
        """
        if not candidates:
            return [], {}

        # Sort candidates descending by base score
        available = sorted(candidates, key=lambda x: x["base_score"], reverse=True)

        selected: list[dict[str, Any]] = []
        selected_archetypes: set[str] = set()
        target_archetypes = set(CORE_ARCHETYPES)

        while len(selected) < self.council_size and available:
            best_idx = -1
            best_mmr = -999.0
            best_candidate = None

            # Check which archetypes are still missing from the council
            unrepresented = target_archetypes - selected_archetypes

            for idx, cand in enumerate(available):
                # Calculate MMR score
                if not selected:
                    mmr = cand["base_score"]
                else:
                    max_sim = max(
                        jaccard_similarity(cand["ngrams"], s["ngrams"])
                        for s in selected
                    )
                    mmr = self.lambda_mmr * cand["base_score"] - (
                        1.0 - self.lambda_mmr
                    ) * max_sim

                # Archetype coverage guardrail:
                # If we still have unrepresented archetypes, only consider candidates from them
                if unrepresented:
                    if cand.get("archetype") in unrepresented:
                        if mmr > best_mmr:
                            best_mmr = mmr
                            best_candidate = cand
                            best_idx = idx
                else:
                    # All core archetypes covered; pick the highest MMR candidate regardless
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_candidate = cand
                        best_idx = idx

            # If for some reason no candidate matched unrepresented archetypes, fall back to best overall MMR
            if best_candidate is None and available:
                for idx, cand in enumerate(available):
                    if not selected:
                        mmr = cand["base_score"]
                    else:
                        max_sim = max(
                            jaccard_similarity(cand["ngrams"], s["ngrams"])
                            for s in selected
                        )
                        mmr = self.lambda_mmr * cand["base_score"] - (
                            1.0 - self.lambda_mmr
                        ) * max_sim
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_candidate = cand
                        best_idx = idx

            if best_candidate is not None:
                cand = available.pop(best_idx)
                cand["mmr_score"] = round(best_mmr, 4)
                cand["selection_order"] = len(selected) + 1
                cand["selection_reason"] = (
                    f"Archetype representation ({cand.get('archetype')})"
                    if cand.get("archetype") in unrepresented
                    else "High composite MMR & stylometric complementarity"
                )
                selected.append(cand)
                if cand.get("archetype"):
                    selected_archetypes.add(cand["archetype"])
            else:
                break

        # Verification of stylometric coverage
        formality_counts = {}
        for p in selected:
            f = p.get("commenting_habits", {}).get("formality", "Unknown")
            formality_counts[f] = formality_counts.get(f, 0) + 1

        metadata = {
            "total_candidates_evaluated": len(candidates),
            "council_size": len(selected),
            "archetype_distribution": {
                arch: sum(1 for p in selected if p.get("archetype") == arch)
                for arch in CORE_ARCHETYPES
            },
            "formality_distribution": formality_counts,
            "lambda_mmr": self.lambda_mmr,
        }

        return selected, metadata

