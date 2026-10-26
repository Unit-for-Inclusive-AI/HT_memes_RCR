# PAN19 English Persona Pool Statistics

This report describes the PAN19 English training data used to construct the 300 simulated English personas in [`artifacts/persona_pool_en.json`](../artifacts/persona_pool_en.json). PAN19 is an author-profiling dataset of Twitter posts. The original author labels are used only to exclude bot accounts; they are not included in persona prompts or treated as persona traits.

## 1. Source Dataset and Selection

| Metric | Count | Notes |
| :--- | ---: | :--- |
| English authors in PAN19 training directory | 4,120 | One XML file per author; 100 documents per author |
| Authors in `truth-train.txt` | 2,880 | The train partition, excluding dev authors |
| Human authors in `truth-train.txt` | 1,440 | Eligible source population before post-quality filtering |
| Bot authors in `truth-train.txt` | 1,440 | Excluded from persona construction |
| Eligible human authors with at least 6 usable English posts | 1,440 | Minimum history condition |
| Personas selected | 300 | Seeded random sample (`seed=2026`) from eligible human authors |

The test and dev label files were not used. PAN19 XML files contain posts under `./documents/document`; the builder retains text with at least four non-mention/non-URL words, English-script letters, and no Han or Devanagari characters. It requires at least six retained posts per author. Up to six examples are sampled across each user's retained history.

## 2. Pool Size and Post Volume

| Metric | Value |
| :--- | ---: |
| Personas | 300 (`P_EN_0001`–`P_EN_0300`) |
| Unique pseudonymous source-user hashes | 300 |
| Example posts stored | 1,800 (6 per persona) |
| Usable posts found across selected authors | 26,643 |
| Mean usable posts per author | 88.81 |
| Median usable posts per author | 91 |
| Minimum / maximum usable posts | 52 / 100 |
| Mean words per usable post | 17.18 |
| Median words per usable post | 16 |
| Minimum / maximum words per usable post | 4 / 60 |

The pool stores six example posts per persona for downstream prompting. `total_available_comments` records the number of usable posts found in that author's XML file; despite the inherited field name, the PAN19 units are tweets/posts, not replies or comments.

## 3. Assigned Debate-Lens Distribution

The six debate lenses are allocated round-robin for balanced experimental coverage. They are simulation instructions, **not beliefs inferred from the source authors**.

| Assigned lens | Personas | Share |
| :--- | ---: | ---: |
| Supportive / Pro-LGBTQ+ Ally | 50 | 16.67% |
| Critical / Conservative | 50 | 16.67% |
| Cultural / Traditional Norms | 50 | 16.67% |
| Skeptical / Ambiguous Stance | 50 | 16.67% |
| Educator / Factual Explainer | 50 | 16.67% |
| Neutral Observer | 50 | 16.67% |
| **Total** | **300** | **100.00%** |

## 4. Observable Writing-Style Metrics

Style features are computed from all usable posts for each selected author. “Formal” here is only a length-based bucket: an average of 12 or more whitespace-separated words per post. It is not a claim about education or social status.

| Feature | Category | Personas | Share |
| :--- | :--- | ---: | ---: |
| Average post length | Formal bucket (≥12 words) | 283 | 94.33% |
| Average post length | Colloquial bucket (<12 words) | 17 | 5.67% |
| Emoji use | Above 20% of posts | 24 | 8.00% |
| Emoji use | At or below 20% | 276 | 92.00% |
| Emphatic punctuation | Above 25% of posts | 25 | 8.33% |
| Emphatic punctuation | At or below 25% | 275 | 91.67% |

## 5. Reproducibility and Limitations

- Rebuild the same sampled pool with `scripts/02_build_persona_pool_pan19_en.py`; the default seed is `2026`.
- Source author IDs are SHA-256 hashed to 12 hexadecimal characters in the output. Raw IDs and PAN19 gender/bot labels are not copied into persona records.
- This is a 2019 Twitter author-profiling dataset, not a corpus of meme replies or anti-LGBTQ+ discussions. Personas provide English-language posting examples and controlled debate lenses; they should not be described as a representative sample of English-speaking meme users.
- Preserve PAN19 attribution and check the source terms before redistributing tweet text or derived records outside this research workspace.
