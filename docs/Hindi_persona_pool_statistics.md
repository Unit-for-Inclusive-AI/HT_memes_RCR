# Koo Hindi Persona Pool Statistics

This report describes the provisional Hindi persona pool in [`artifacts/persona_pool_hi_koo_provisional_300.json`](../artifacts/persona_pool_hi_koo_provisional_300.json). The pool contains a controlled sample of Koo creators and their historical posts; it is not a representative sample of Hindi-speaking social-media users. Language labels are automatic and have not been checked by a Hindi-speaking reviewer.

## 1. Source and Selection

The source is `Posts/hi_posts.json` in the Koo posts archive stored under `data/external/Koo/`. The published Koo dataset reports 35,074,082 Hindi posts and 552,941 Hindi user profiles across its full release ([Mekacher et al., 2024](https://ojs.aaai.org/index.php/ICWSM/article/view/31442)). Those source-dataset totals are not the size of this persona pool.

The builder streamed all 35,074,082 records in the local Hindi-locale partition. It selected candidate creators deterministically using a 2% hash-based rate, then retained posts with Devanagari and Hindi lexical evidence, excluding contact details and account references. Within that sampled creator set, 8,971 creators had at least one post passing those text filters, and 3,571 had at least five. A deterministic sample of 1,000 eligible creators was selected for automatic language identification.

Lingua 2.2.0 labeled the 5,909 sampled posts. A post was retained as Hindi when Hindi was the top result from the Hindi/Marathi detector with confidence of at least 0.65 and all-language segment detection found no segment outside Hindi and Marathi. Code-mixed, Marathi, and unclear posts were not retained. Of the 1,000 creators, 525 had at least five posts labeled Hindi. The final pool selects 50 creators for each of the six assigned debate lenses from those eligible creators. The selection and labeling script is [`scripts/05_filter_persona_pool_koo_hi.py`](../scripts/05_filter_persona_pool_koo_hi.py); it calls [`scripts/04_build_persona_pool_koo_hi.py`](../scripts/04_build_persona_pool_koo_hi.py) for Koo post sampling.

## 2. Pool Size and Post Volume

| Metric | Value |
| :--- | ---: |
| Personas retained | 300 |
| Unique pseudonymous source-user hashes | 300 |
| Example posts included | 1,640 |
| Personas with five included posts | 160 (53.33%) |
| Personas with six included posts | 140 (46.67%) |
| Mean / median posts per persona | 5.47 / 5 |
| Mean post length | 165.98 characters |
| Median post length | 130 characters |
| Minimum / maximum post length | 12 / 606 characters |

The 1,640 posts are sampled examples retained after automatic language filtering; they are not the creators' complete Koo histories.

## 3. Assigned Debate-Lens Distribution

The final pool is balanced across six debate lenses. Lenses are assigned simulation instructions and are **not beliefs inferred from source authors**.

| Assigned lens | Personas | Share |
| :--- | ---: | ---: |
| Supportive / Pro-LGBTQ+ Ally | 50 | 16.67% |
| Critical / Conservative | 50 | 16.67% |
| Cultural / Traditional Norms | 50 | 16.67% |
| Skeptical / Ambiguous Stance | 50 | 16.67% |
| Educator / Factual Explainer | 50 | 16.67% |
| Neutral Observer | 50 | 16.67% |
| **Total** | **300** | **100.00%** |

## 4. Automatic Language Labels and Observable Text Metrics

The automatic-label file [`artifacts/koo_hi_persona_language_auto_labels_1000.csv`](../artifacts/koo_hi_persona_language_auto_labels_1000.csv) includes all labeled candidate posts, the top two language results and confidence values, and a blank manual-review field. Its labels are distributed as follows:

| Automatic label | Candidate posts | Share |
| :--- | ---: | ---: |
| Hindi | 4,322 | 73.14% |
| Code-mixed | 644 | 10.90% |
| Marathi | 79 | 1.34% |
| Unclear | 864 | 14.62% |
| **Total** | **5,909** | **100.00%** |

For the 1,640 posts included in the final pool, the Hindi top-label confidence has a mean of 0.900, a median of 0.935, and a range of 0.650–1.000. A total of 958 posts (58.41%) have confidence of at least 0.90. These are model confidence scores, not human verification.

| Observable feature in included posts | Posts | Share |
| :--- | ---: | ---: |
| Contains an emoji | 425 | 25.91% |
| Contains repeated emphatic punctuation | 245 | 14.94% |

Post length is counted in Unicode characters. Emoji presence uses the builder's Unicode emoji range; emphatic punctuation counts repeated `!` or `?`, three or more periods, or repeated ellipses. These are surface features of the stored text, not demographic or psychological traits. The builder's `formality` field is fixed and is not reported as an observed statistic.

## 5. Reproducibility and Limitations

- Install the optional detector with `python -m pip install -r requirements-language-detection.txt`, then run `python scripts/05_filter_persona_pool_koo_hi.py`. Defaults reproduce the 1,000-creator candidate sample, six examples per creator, five-post minimum, 2% deterministic creator rate, seed 2026, and a balanced 300-person output.
- The initial 50-person pilot and its 26-person automatically filtered pool are retained as earlier artifacts. The 300-person pool supersedes them for the Hindi pilot runner.
- The language partition is not guaranteed to contain only Hindi. Lingua can misclassify short posts, named entities, hashtags, and code-mixed writing; no Hindi-speaking reviewer has completed a manual check.
- The creator sample is drawn from a 2% deterministic hash sample, then filtered by text and language rules. Results may not generalize to all Koo creators or Hindi users. Koo is a particular historical platform, and the published study describes the platform as having political and community biases.
- Creator IDs are hashed to 12 hexadecimal characters before inclusion. Check the source dataset's terms and attribution requirements before redistributing post text or derived records beyond this research workspace.
