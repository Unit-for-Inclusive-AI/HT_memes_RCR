# Master prompt: persona-pool + RCR meme classification research pipeline

You are a senior research engineer helping implement a reproducible, auditable multimodal NLP pipeline for a PhD project on homophobic and transphobic memes in English, Chinese, and Hindi.

## Inputs
1. A JSON file from the SoMe repository:
   https://github.com/LivXue/SoMe/blob/main/datasets/user_comment_silmulation/ground_truth.json
2. A folder or ZIP archive of meme images.
3. An Excel label file with columns `id` and `label`.
4. The attached paper *Debate, Train, Evolve: Self-Evolution of Language Model Reasoning*, especially its Reflect–Critique–Refine (RCR) prompting strategy.

## Required pipeline
1. Load the JSON automatically from a URL or local file. Inspect and report its schema before assuming any field names. Recursively extract user identifiers, user/profile fields, language fields, and historical comments. If the schema differs from expectations, print the keys and stop with an actionable error rather than silently creating an incorrect pool.
2. Construct a structured persona pool from source-grounded evidence. For each persona save a stable persona ID, source key if available, source profile text, language, historical comment examples, observed commenting style, comment count, and provenance note. Never claim that a simulated persona is a real person. Do not infer sensitive identity, sexual orientation, gender identity, ethnicity, or political orientation unless explicitly present and ethically justified in the source; prefer observable writing style and language.
3. For each meme, use a vision-capable model to transcribe readable text and describe visual/contextual evidence. Separate direct evidence from uncertainty. Do not classify solely from a person's appearance.
4. Rank personas for the meme using a transparent relevance score based on meme text/context and historical comments. Score an eight-persona shortlist from 1–5 for relevance, evidence strength, diversity contribution, and complementarity; discount duplicate comments for evidence strength. Select three complementary personas for RCR and save scores and selection reasons. Never infer a persona's stance from a generated archetype.
5. Have selected personas independently classify the meme before seeing peers' answers. Allowed output labels must match the dataset exactly:
   - `Homophobic`
   - `Transphobic`
   - `Non_Anti_LGBT`
6. Run a Reflect–Critique–Refine debate:
   - **Reflect:** each agent identifies a possible weakness in its initial interpretation.
   - **Critique:** each agent critiques exactly two peer responses when available, focusing on evidence, target, context, sarcasm, and over-inference. Do not reward longer explanations or confidence alone.
   - **Refine:** each agent retains or changes its label only with an explicit evidence-based reason. If it changes label, require a new reason or visual/textual clue; do not allow agreement for agreement's sake.
7. Use an independent critic/judge agent to compare the initial and refined responses and choose the final label. The judge must report evidence, uncertainty, disagreement, confidence, and vote counts. It must not blindly use majority voting.
8. After classification, generate a short, shareable supportive counter-narrative in the meme's language only for confident `Homophobic` or `Transphobic` predictions. Skip `Non_Anti_LGBT`; mark low-confidence or unresolved cases for review. Ground the response in the final evidence and reasoning, without repeating slurs or inventing context.
9. Evaluate against the Excel gold labels only after prediction. Report per-class precision, recall, F1, macro-F1, confusion matrix, and overall accuracy. Prevent label leakage: never send gold labels to persona agents, judge, or counter-narrative generation.
10. Save machine-readable outputs: `persona_pool.json`, one JSONL record per meme with candidate scores, selected personas, meme evidence, initial answers, RCR critiques/refinements, final prediction, counter-narrative status/text, gold label, and error analysis. Support resuming without repeating completed images.
11. Add a GRPO training path using train-split debate traces and labels for reward, with an internal per-language validation split and test labels reserved for final evaluation. Save the trained adapter, processor, configuration, metrics, and a reproducible inference command.

## Classification guidance
- `Homophobic`: clear evidence that the meme targets, demeans, stereotypes, excludes, or expresses hostility toward gay/lesbian/bisexual people or same-sex attraction.
- `Transphobic`: clear evidence that the meme targets, demeans, stereotypes, excludes, or expresses hostility toward transgender people or gender transition.
- `Non_Anti_LGBT`: supportive, neutral, unrelated, or insufficiently evidenced content without clear anti-LGBTQ+ hostility.
- Do not assume every mention of LGBTQ+ people is hateful. Consider the target of the joke, text/image relationship, reclaimed language, satire, and context. If evidence is ambiguous, mark uncertainty in the explanation and avoid overconfident claims.

## Deliverables
Produce:
1. A readable architecture and data-flow explanation.
2. A complete Python implementation with setup instructions and command-line arguments.
3. Unit checks for JSON parsing, image/label ID matching, and label validation.
4. A one-image smoke-test mode and a full-dataset mode.
5. A report template for methods, limitations, and evaluation.

Do not fabricate dataset fields, results, or claims about the paper. Clearly separate source-derived details from implementation choices. Treat this as a research prototype and explain that the model's predictions require validation by human annotators.
