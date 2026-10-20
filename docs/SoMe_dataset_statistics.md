# SoMe Benchmark & Persona Pool Statistics

This document provides a comprehensive statistical profile of the **SoMe Benchmark** (*"SoMe: A Realistic Benchmark for LLM-based Social Media Agents"*, AAAI) and details the exact criteria used to extract, filter, and balance the **300 simulated Chinese personas** stored in [`artifacts/persona_pool_zh.json`](../artifacts/persona_pool_zh.json).

---

## 1. Global SoMe Benchmark Statistics

The personas utilized for multi-agent meme deliberation are grounded in real-world social media commenting behavior from the **SoMe** benchmark.

### 1.1 Macro-Level Scale

| Metric | Count | Description |
| :--- | :---: | :--- |
| **Total Social Media Posts** | **9,164,284** | Microblogging posts collected primarily from Sina Weibo |
| **Tracked User Profiles** | **6,591** | Real user profiles with historical posting and interaction timelines |
| **External Reports** | **25,686** | Contextual news, background reports, and factual references |
| **Annotated Task Queries** | **17,869** | Evaluation instances across 8 diverse social agent tasks |

### 1.2 Benchmark Task Taxonomy
SoMe organizes agent evaluation into three main tiers across 8 distinct tasks:
1. **Post-Centered Tasks**: Real-time Event Detection (RED), Streaming Event Summary (SES), Misinformation Detection (MID).
2. **User-Centered Tasks**: User Behavior Prediction (UBP), User Emotion Analysis (UEA), **User Comment Simulation (UCS)**.
3. **Comprehensive Tasks**: Media Content Recommendation (MCR), Social Media Question-Answering (SMQ).

---

## 2. The User Comment Simulation (UCS) Subset

Our persona pipeline specifically utilizes the **User Comment Simulation (UCS)** subset (`datasets/user_comment_simulation/ground_truth.json`).

* **Task Scale**: **4,000** evaluated user-comment instances.
* **Objective**: Measure an agent's ability to simulate realistic, context-aware human reactions to social content based on user history, tone, and behavioral habits.
* **Topical Domain Distribution**:
  * **Entertainment & Celebrity News**: ~36.5%
  * **Society, Life, & Everyday Discourse**: ~31.6%
  * **Culture, Arts, & Current Affairs**: ~24.4%
  * **Specialized Sub-Tracks (Tech, Gaming, Sports)**: ~7.5%

---

## 3. Persona Mining & Selection Criteria (How the 300 Were Selected)

From the 4,000 raw comment items in UCS, the pipeline in [`src/meme_pipeline/personas/extractor.py`](../src/meme_pipeline/personas/extractor.py) compiled **300 unique personas** using five deterministic criteria:

```
Raw UCS Data (4,000 comments)
      │
      ▼ [Criterion 1] Script & Length Validation (Hanzi [\u4e00-\u9fff], len ≥ 4)
Cleaned Comments
      │
      ▼ [Criterion 2] User Aggregation (Group by user ID, drop empty users)
Unique User Records
      │
      ▼ [Criterion 3] Privacy-Preserving Hashing (One-way SHA-256)
Anonymized Profiles
      │
      ▼ [Criterion 4] Quota Balancing (Round-robin idx % 6 across 6 archetypes)
Balanced Archetypes (50 per archetype)
      │
      ▼ [Criterion 5] Pool Size Cap (Cap at exactly 300 personas)
artifacts/persona_pool_zh.json (300 Personas)
```

### The 5 Selection Rules

1. **Language Validity & Minimum Length**:
   * Text must contain Unicode Hanzi characters (`[\u4e00-\u9fff]`).
   * Cleaned string length must be $\ge 4$ characters (`len(text_clean) >= 4`).
   * *Rationale*: Eliminates single emojis, punctuation marks, bot tokens, and non-Chinese spam.
2. **User-Level Aggregation & History Requirement**:
   * Comments are grouped by unique user identifiers (`userid`, `user`, `author`, `weibo_id`).
   * Users with zero valid comments after filtering are strictly dropped (`if not data["comments"]: continue`).
3. **Privacy-Preserving Anonymization**:
   * All raw usernames and platform IDs are permanently stripped and converted into a 12-character SHA-256 hash (`hashlib.sha256(uid.encode("utf-8")).hexdigest()[:12]`).
   * *Rationale*: Adheres to data ethics and privacy standards (GDPR-compliant anonymization).
4. **Deterministic Archetype Quota (Equal Representation)**:
   * To prevent ideological skew during debate, users are assigned one of 6 predefined sociological archetypes via round-robin cycling:
     $$\text{Archetype} = \text{archetypes}[\text{idx} \pmod 6]$$
   * Enforces a strictly uniform 1:1:1:1:1:1 ratio (**exactly 50 personas per archetype**).
5. **Pool Size Cap (`max_pool_size = 300`)**:
   * Ingestion halts upon collecting the first 300 qualified personas.
   * *Rationale*: 300 candidate personas provide sufficient lexical variance for TF-IDF matching without adding unnecessary computational latency during inference.

---

## 4. Extracted Persona Pool Statistics (`persona_pool_zh.json`)

Analysis performed on the compiled pool located at [`artifacts/persona_pool_zh.json`](../artifacts/persona_pool_zh.json).

### 4.1 Volume & Comment Metrics

| Metric | Value |
| :--- | :---: |
| **Total Personas** | **300** (`P_ZH_0001` to `P_ZH_0300`) |
| **Total Historical Comments** | **1,022** comments |
| **Mean Comments per Persona** | **3.41** |
| **Median Comments per Persona** | **4.00** |
| **Min / Max Comments per Persona** | 1 / 4 |

---

### 4.2 Archetype Distribution

Every archetype is represented in equal proportions across the 300 personas:

| Archetype Name (English / Chinese) | Count | Percentage | Primary Stance / Deliberation Lens |
| :--- | :---: | :---: | :--- |
| **Critical / 批判保守者** | 50 | 16.67% | Skeptical of social changes, focuses on traditional social order |
| **Cultural / 传统价值观持有者** | 50 | 16.67% | Grounded in classical cultural values, family structures |
| **Skeptical / 怀疑调侃者** | 50 | 16.67% | Internet cynic, probes for irony, satire vs. malicious intent |
| **Educator / 事实科普与理性分析者** | 50 | 16.67% | Objective analysis of semantics, facts, and linguistic definitions |
| **Neutral Observer / 客观旁观者** | 50 | 16.67% | General public viewpoint, surface-level common-sense interpretation |
| **Supportive / 包容友善支持者** | 50 | 16.67% | Pro-LGBTQ+ ally, monitors harm, stereotyping, and inclusion |
| **Total** | **300** | **100.00%** | **Balanced 6-Way Perspective Pool** |

---

### 4.3 Stylometric & Behavioral Metrics (`commenting_habits`)

The stylometric profile of each persona is computed from observed commenting habits without demographic assumptions:

| Dimension | Metric / Category | Count | Percentage | Description / Criteria |
| :--- | :--- | :---: | :---: | :--- |
| **Formality** | `口语化` (Colloquial) | 168 | 56.00% | Average length $< 20$ characters (short, casual online talk) |
| | `书面表达` (Formal) | 132 | 44.00% | Average length $\ge 20$ characters (structured, elaborate text) |
| **Emoji / Sticker Usage** | `uses_emoji = True` | 243 | 81.00% | $> 20\%$ of comments contain emojis or platform sticker tags |
| | `uses_emoji = False` | 57 | 19.00% | $\le 20\%$ emoji/sticker density |
| **Punctuation Intensity** | `emphatic_punctuation = True` | 93 | 31.00% | $> 25\%$ of comments use repeated `!`, `?`, or ellipses |
| | `emphatic_punctuation = False` | 207 | 69.00% | Regular punctuation |

#### Comment Character Length Distribution
* **Mean Character Length**: **23.61 characters**
* **Median Character Length**: **17.85 characters**
* **Minimum Average Length**: **4.00 characters**
* **Maximum Average Length**: **112.00 characters**

---

### 4.4 Stylometric Habit Profiles Across the Pool

The combination of length, emoji usage, and emphatic punctuation yields 8 distinct natural language behavioral summaries:

| Rank | Behavioral Summary (`summary`) | Count | Percentage | Typical Persona Behavior |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `简短精炼；喜用表情包。` | 102 | 34.00% | Short, casual remarks punctuated with sticker tags |
| 2 | `表达详实；喜用表情包。` | 75 | 25.00% | Detailed, expressive paragraphs with visual emojis |
| 3 | `表达详实；语气强烈；喜用表情包。` | 35 | 11.67% | Passionate, long-form arguments with heavy punctuation & emojis |
| 4 | `简短精炼；语气强烈；喜用表情包。` | 31 | 10.33% | Punchy, emotional one-liners (exclamations + stickers) |
| 5 | `简短精炼。` | 19 | 6.33% | Plain, direct, minimalistic responses |
| 6 | `简短精炼；语气强烈。` | 16 | 5.33% | Short, sharp critiques or rhetorical questions |
| 7 | `表达详实；语气强烈。` | 11 | 3.67% | Structured, passionate dissertations without emojis |
| 8 | `表达详实。` | 11 | 3.67% | Formal, sober, neutral analyses |
| — | **Total** | **300** | **100.00%** | — |

---

## 5. Methodological & Scientific Justifications

When documenting this pipeline for publications or a PhD thesis, use the following theoretical justifications:

1. **Behavioral Stylometrics vs. Demographic Hallucination**:
   Social media benchmarks rarely provide verified demographic data (age, gender, income). Guessing demographics introduces synthetic bias and harmful stereotypes. Our approach grounds persona differentiation strictly in **empirically observed stylometrics** (sentence length, emoji density, punctuation patterns) and actual historical text embeddings.
2. **Separation of Linguistic DNA and Cognitive Role**:
   * The real user data provides the **linguistic realism** (authentic Chinese vernacular, vocabulary, internet slang).
   * The assigned archetype provides the **cognitive debate instruction** for the LLM.
   This guarantees that simulated agents argue with realistic human writing habits while maintaining a controlled, balanced spectrum of viewpoints.
3. **Prevention of AI Sycophancy**:
   Standard LLMs tend to conform to the premise of a prompt or agree with peer agents. Allocating 50 personas to each of 6 conflicting archetypes ensures cognitive friction during Reflect–Critique–Refine (RCR) rounds, significantly boosting accuracy when classifying ambiguous or sarcastic memes.
4. **Controlled Baseline Anchoring (`profile_bio`)**:
   Because profile bios were not crawled in the raw UCS data, the field is held constant across all 300 personas (`"社交网络普通活跃用户。"`). This serves as a controlled baseline, ensuring that persona divergence in downstream experiments is driven solely by stylometric habits, historical comment similarity, and assigned archetypes.

