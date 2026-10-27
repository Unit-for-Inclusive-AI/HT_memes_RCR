# Persona DTE meme classification

This repository builds evidence-based persona shortlists from SoMe comments,
selects a complementary trio for meme classification, records Reflect-Critique-
Refine (RCR) debate traces, and generates supportive counter-narratives for
confident anti-LGBTQ+ classifications. A separate GPU training path uses GRPO
to fine-tune a Qwen2.5-VL adapter from training-split debate traces.

See [TRAINING.md](TRAINING.md) for commands to generate Chinese, English, and
Hindi traces, install the optional GPU training stack, train/evaluate an adapter,
and run final inference.

## Documentation

- [docs/dataset_statistics.md](docs/dataset_statistics.md): Multi-language statistics, split compositions, and label distributions for the raw HT Meme dataset (Chinese, English, Hindi).
- [docs/some_dataset_statistics.md](docs/some_dataset_statistics.md): Benchmark statistics for SoMe, along with mining criteria, stylometrics, and archetype distributions for the 300 Chinese personas.
- [docs/English_persona_pool_statistics.md](docs/English_persona_pool_statistics.md): PAN19 source and persona-pool statistics for English.
- [docs/Hindi_persona_pool_statistics.md](docs/Hindi_persona_pool_statistics.md): Source, filtering, and provisional pool statistics for the Koo-derived Hindi personas.
- [docs/persona_council_statistics.md](docs/persona_council_statistics.md): Pruned 8-member Chinese Persona Council selected for Stage 1 offline optimization.
- [docs/dte_rcr_workflow.md](docs/dte_rcr_workflow.md): Theoretical alignment with Debate-Train-Evolve (DTE - EMNLP 2025) and complete RCR + GRPO training workflow.

The persona records are simulations derived from historical comments. Their
archetypes are not treated as evidence of a source user's stance. Dataset labels
are used for training rewards and evaluation, never in persona or counter-
narrative prompts.
