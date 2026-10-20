# Persona-DTE: Multi-Agent RCR Debate & GRPO Self-Evolution Workflow

This document details the architecture, theoretical alignment, and operational procedures for our multimodal anti-LGBTQ+ meme classification pipeline, grounded in the EMNLP 2025 research paper:
> **"DEBATE, TRAIN, EVOLVE: Self-Evolution of Language Model Reasoning"** (Gaurav Srivastava, Zhenyu Bi, Meng Lu, Xuan Wang — Virginia Tech)  
> *EMNLP 2025 Main* | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1666.pdf) | [Official GitHub](https://github.com/ctrl-gaurav/Debate-Train-Evolve)

---

## 1. Theoretical Foundations & Architectural Mapping

The core thesis of the **DTE (Debate, Train, Evolve)** framework is that language models can self-evolve complex reasoning capabilities without requiring human-annotated reasoning chains. Instead, models deliberate through a structured multi-agent debate protocol, collect high-quality reasoning traces, and fine-tune a student model using reinforcement learning (RL).

### How Our Pipeline Adapts and Extends DTE

| Framework Dimension | Original DTE Paper (Gaurav et al., 2025) | Our Persona-DTE Extension |
| :--- | :--- | :--- |
| **Domain & Modality** | Text-only math & logic (GSM8k, MATH, Big-Bench) | **Multimodal** Anti-LGBTQ+ Hate Meme Detection (Vision + Text) in Chinese, English, Hindi |
| **Agent Heterogeneity** | Homogeneous generic agents (identical prompting) | **Culturally Grounded Personas** ([`artifacts/persona_council_zh.json`](../artifacts/persona_council_zh.json)) with contrasting societal stances |
| **Debate Protocol** | **Reflect–Critique–Refine (RCR)** | **Domain-Specific RCR**: Self-critique of bias, peer critique of visual evidence/sarcasm, and label refinement |
| **Consensus / Adjudication** | Majority voting or LLM Judge | **Independent Critic Judge**: Evidence-weighted judicial arbiter preventing sycophancy |
| **Student Fine-Tuning** | RL / Direct Preference Optimization on debate trajectories | **GRPO (Group Relative Policy Optimization)** fine-tuning a vision-language model (`Qwen2.5-VL`) |
| **Social Intervention** | None (classification only) | **Supportive Counter-Narrative Generation**: Empathetic counterspeech for detected hate memes |

---

## 2. End-to-End Pipeline Workflow

The workflow consists of three interconnected stages:

```
[STAGE 1: DEBATE & TRACE GENERATION]
Meme Image + Label Workbook
          │
          ▼
   summarize_meme() ➔ Vision transcription & cultural context clues
          │
          ▼
   select_personas() ➔ 1–5 Scoring (Relevance, Strength, Diversity, Complementarity)
          │            Selects optimal Top-3 debaters from the 8-member Council
          ▼
   Multi-Agent RCR Debate
     ├── Round 1 (Initial): Independent classification per persona
     ├── Round 2 (Reflect & Critique): Self-bias reflection + 2 peer critiques
     └── Round 3 (Refine): Evidence-grounded position update
          │
          ▼
   critic_judge() ➔ Final consensus verdict (Homophobic / Transphobic / Non_Anti_LGBT)
          │
          ▼ (If hateful)
   generate_counter_narrative() ➔ Supportive Chinese counterspeech
          │
          ▼
artifacts/chinese_council_gemini_debate.jsonl (Recorded Traces)
          │
          ├─────────────────────────────────────────────┐
          ▼                                             ▼
[STAGE 2: TRAIN (GRPO)]                      [STAGE 3: EVOLVE & INFERENCE]
scripts/train_grpo_vlm.py                     Deploy Promoted PEFT Adapter
Qwen2.5-VL Adapter Fine-Tuning                Evaluate on Unseen Test Splits
Reward: Accuracy (1.0) + Format (0.1)         Generate Final Performance Metrics
```

---

## 3. The Reflect-Critique-Refine (RCR) Protocol

Implemented in [`Code/meme_persona_pipeline.py`](../Code/meme_persona_pipeline.py) and [`src/meme_pipeline/debate/prompts.py`](../src/meme_pipeline/debate/prompts.py):

### Round 1: Initial Independent Classification
Each of the 3 selected debaters independently inspects the meme and produces a structured verdict using its unique persona profile:
- **Inputs**: Visual description, text transcription, persona archetype, writing style, and historical comment context.
- **Output Schema**:
  ```json
  {
    "label": "Homophobic" | "Transphobic" | "Non_Anti_LGBT",
    "confidence": 0.85,
    "evidence": "Concrete textual and visual cues",
    "ambiguity": "Uncertain cultural aspects",
    "rationale": "Step-by-step reasoning from persona lens",
    "persona_id": "P_ZH_0017"
  }
  ```

### Round 2 & 3: Reflect, Critique, and Refine
Agents see the initial interpretations of their peers and execute the three core RCR phases:
1. **Reflect**: Identify vulnerabilities or potential subjective assumptions in their own initial assessment.
2. **Critique**: Objectively evaluate two peers. Check whether peers overlooked sarcasm, misidentified reclaimed slang, or over-inferred malicious intent.
3. **Refine**: Update or confirm their decision. An agent is **forbidden** from changing its stance merely to conform; it must cite new textual or visual evidence.

### Round 4: Independent Critic Judge
An impartial judge agent synthesizes the entire deliberation trace:
- Compares initial and refined answers.
- Rewards evidence quality and contextual groundedness over simple majority votes.
- Records disagreement summaries and confidence metrics.

---

## 4. Stage 2: GRPO Training Details (`scripts/train_grpo_vlm.py`)

The collected RCR debate traces serve as offline training data to fine-tune a multimodal base model (`Qwen2.5-VL-3B-Instruct`) via **Group Relative Policy Optimization (GRPO)**:

### Reward Engineering
During GRPO rollouts, the trainer samples 8 completions per meme prompt and applies three composite reward functions ([`src/meme_pipeline/training/rewards.py`](../src/meme_pipeline/training/rewards.py)):

$$\text{Reward}(y) = R_{\text{accuracy}}(y, y^*) + R_{\text{format}}(y) + R_{\text{evidence}}(y)$$

1. **Classification Accuracy Reward ($R_{\text{accuracy}} = 1.0$)**:
   - Compares the model's predicted label against the verified training workbook gold label.
2. **Structured Output Reward ($R_{\text{format}} = 0.1$)**:
   - Verifies valid JSON containing all mandatory keys (`label`, `confidence`, `evidence`, `rationale`).
3. **Evidence Grounding Reward ($R_{\text{evidence}} = 0.1$)**:
   - Ensures the evidence field is non-empty, contains meaningful explanation, and does not repeat raw label tokens.

### Promotion Criterion
The trainer splits traces into internal train and validation splits. The trained PEFT LoRA adapter is only **promoted** for final test inference if its validation Macro-F1 strictly outperforms the base model.

---

## 5. Execution Commands

### Step 1: Generate Debate Traces with the Curated Council
Run RCR debate on the training split using the 8-member Chinese Persona Council:

```bash
# Using Gemini on Vertex AI:
bash scripts/run_chinese_gemini_debate.sh

# Or using OpenAI API:
bash scripts/run_chinese_gpt_debate.sh
```

### Step 2: Train the Vision-Language Adapter with GRPO
On a CUDA-enabled GPU host:

```bash
# Install training dependencies
pip install -r requirements.txt -r requirements-training.txt

# Run GRPO training
python scripts/train_grpo_vlm.py \
  --traces artifacts/chinese_train_10_gemini_debate.jsonl \
  --data-root data/raw \
  --output-dir artifacts/grpo_meme_model \
  --epochs 3
```

### Step 3: Final Inference & Counter-Narratives on Test Split
Deploy the promoted adapter to evaluate unseen test memes:

```bash
python Code/meme_persona_pipeline.py \
  --backend local \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter-path artifacts/grpo_meme_model/final_adapter \
  --language Chinese \
  --split test \
  --images-zip data/raw/Chinese/Test/Test_images \
  --labels-xlsx data/raw/Chinese/Test_labels.xlsx \
  --output artifacts/final_test_predictions_zh.jsonl \
  --limit 0
```

