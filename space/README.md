---
title: RELATE — The Embedding Knew More
emoji: 🧭
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.22.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
models:
  - microsoft/codebert-base
---

# RELATE demo

This Space demonstrates relation-aware search over frozen CodeBERT embeddings.

It compares two candidate Python functions against a query using:

- raw cosine distance;
- a frozen ridge projection into three predicted AST coordinates;
- the corresponding true AST-coordinate distance.

The projection was reconstructed from the preserved Option B training cache and
verified against the published 4,000-row test prediction array before export.
Submitted Python is parsed and embedded but never executed.

## Historical result

| Method | Hard-negative ordering accuracy |
|---|---:|
| Raw cosine | `0.532458984375` |
| Raw Euclidean | `0.533314453125` |
| Predicted relation executor | `0.7328515625` |

This demo does not reopen RELATE-E01, which remains `EXPERIMENT_INVALID`.
