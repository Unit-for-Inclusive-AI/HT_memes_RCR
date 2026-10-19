from __future__ import annotations

import unittest

from meme_pipeline.counter_narratives import (
    counter_narrative_status,
    generate_counter_narrative,
    has_unresolved_uncertainty,
)
from meme_pipeline.personas.selector import select_personas
from meme_pipeline.training.data import make_prompt, split_train_validation
from meme_pipeline.training.rewards import (
    classification_accuracy_reward,
    evidence_reward,
    structured_output_reward,
)


def persona(persona_id: str, comments: list[str], style: str, language: str = "Chinese"):
    return {
        "persona_id": persona_id,
        "language": language,
        "commenting_style": style,
        "historical_comments": comments,
        "archetype": "fabricated archetype must not affect scoring",
    }


class PersonaSelectionTests(unittest.TestCase):
    def test_scores_candidates_and_selects_distinct_trio(self):
        pool = [
            persona("p1", ["这张图片里有针对跨性别者的刻板印象。"] * 4, "brief"),
            persona("p2", ["这张图片里有针对跨性别者的刻板印象。"], "brief"),
            persona("p3", ["图中文字需要结合视觉语境判断。"], "careful"),
            persona("p4", ["我关注图片的文字和表达对象。"], "analytical"),
            persona("p5", ["这个表达可能是反讽。"], "casual"),
        ]
        candidates, selected = select_personas(
            pool, "跨性别者刻板印象", shortlist_size=4, final_count=3
        )
        self.assertEqual(len(candidates), 4)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({item["persona_id"] for item in selected}), 3)
        for item in candidates:
            self.assertEqual(set(item["persona_scores"]), {
                "relevance", "evidence_strength", "diversity_contribution", "complementarity"
            })
            self.assertTrue(all(1 <= score <= 5 for score in item["persona_scores"].values()))
        counts = {item["persona_id"]: item["unique_comment_count"] for item in candidates}
        self.assertEqual(counts["p1"], 1)

    def test_small_pool_returns_available_count(self):
        candidates, selected = select_personas(
            [persona("only", ["hello"], "brief")], "hello", shortlist_size=8, final_count=3
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(selected), 1)


class CounterNarrativeTests(unittest.TestCase):
    def test_routes_by_label_confidence_and_uncertainty(self):
        self.assertEqual(counter_narrative_status({"label": "Non_Anti_LGBT"}), "not_applicable")
        self.assertEqual(counter_narrative_status({
            "label": "Homophobic", "confidence": 0.69, "uncertainty": "none"
        }), "needs_review")
        self.assertEqual(counter_narrative_status({
            "label": "Transphobic", "confidence": 0.9, "uncertainty": "text is unclear"
        }), "needs_review")
        self.assertEqual(counter_narrative_status({
            "label": "Transphobic", "confidence": 0.9, "uncertainty": "No uncertainty detected"
        }), "generated")
        self.assertFalse(has_unresolved_uncertainty("No uncertainty detected"))

    def test_generates_grounded_language_specific_response_only_when_eligible(self):
        called = []
        result = generate_counter_narrative(
            {"label": "Homophobic", "confidence": 0.9, "uncertainty": "none",
             "evidence": ["caption stereotypes gay men"], "final_reason": "The caption targets gay men."},
            {"visible_text": "..."}, "English",
            lambda prompt: called.append(prompt) or '{"text":"Respect and dignity matter.","language":"English"}',
        )
        self.assertEqual(result["status"], "generated")
        self.assertEqual(result["language"], "English")
        self.assertIn("Respect and dignity", result["text"])
        self.assertIn("The caption targets gay men", called[0])
        skipped = generate_counter_narrative(
            {"label": "Non_Anti_LGBT"}, {}, "Chinese", lambda _: self.fail("must not call generator")
        )
        self.assertEqual(skipped["status"], "not_applicable")


class RewardAndDataTests(unittest.TestCase):
    def test_grpo_rewards_cover_correctness_format_and_evidence(self):
        good = '{"label":"Homophobic","confidence":0.8,"evidence":"caption targets a group"}'
        malformed = "not json"
        self.assertEqual(classification_accuracy_reward([good, malformed], ["Homophobic", "Transphobic"]), [1.0, 0.0])
        self.assertEqual(structured_output_reward([good, malformed]), [0.1, 0.0])
        self.assertEqual(evidence_reward([good, '{"label":"Homophobic","evidence":" "}']), [0.1, 0.0])

    def test_training_prompt_excludes_gold_labels(self):
        row = {
            "language": "Chinese",
            "gold_label": "Transphobic",
            "meme_analysis": {"visible_text": "example"},
            "initial_debate": [{"persona_id": "p1", "label": "Transphobic", "evidence": "text"}],
            "rcr_refined_responses": [{"persona_id": "p1", "label": "Transphobic", "refinement_reason": "context"}],
        }
        prompt = str(make_prompt(row))
        self.assertNotIn('"gold_label"', prompt)

    def test_validation_split_is_reproducible_and_keeps_language_groups(self):
        rows = [
            {"image_id": i, "language": language,
             "gold_label": "Homophobic" if i % 2 else "Non_Anti_LGBT"}
            for language in ("Chinese", "English") for i in range(20)
        ]
        train, validation = split_train_validation(rows, 0.2, 19)
        train_again, validation_again = split_train_validation(rows, 0.2, 19)
        self.assertEqual([r["image_id"] for r in train], [r["image_id"] for r in train_again])
        self.assertEqual([r["image_id"] for r in validation], [r["image_id"] for r in validation_again])
        self.assertTrue(train and validation)
        self.assertEqual({r["language"] for r in validation}, {"Chinese", "English"})


if __name__ == "__main__":
    unittest.main()
