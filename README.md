# RELATE

RELATE is a focused research project asking one question:

> Do frozen embeddings contain relation-specific predictive information that their default cosine or Euclidean geometry underexposes, and can that information be recovered in a way that adds value beyond strong supervised alternatives?

This repository is a clean restart. It does **not** inherit the architecture, workflow, experiment numbering, or project structure of either earlier attempt.

## Source archives

Two previous repositories are retained as historical source archives:

1. [`ernanhughes/similarity_is_relative`](https://github.com/ernanhughes/similarity_is_relative) — the original project. It failed as a sustainable project, but contains one important bounded result: Option B showed that three independently predicted AST-derived coordinates exposed a structural relation substantially better than raw CodeBERT cosine or Euclidean geometry on a preregistered hard-negative evaluation.
2. [`ernanhughes/relate_attempt`](https://github.com/ernanhughes/relate_attempt) — the second attempt. It sharpened the comparison against supervised metric-learning and direct pair models, but RELATE-E01 terminated as `EXPERIMENT_INVALID` when its preregistered shuffled-target control failed. Its primary test was not executed.

The archives are evidence and implementation sources. They are not the active project.

## Scientific state carried forward

### Established narrowly

Option B established this bounded premise:

> In repository-separated real Python code, independently predicted cyclomatic complexity, maximum control nesting depth, and distinct call-site coordinates exposed a frozen three-way structural relation materially better than raw CodeBERT cosine or Euclidean geometry on the registered hard-negative test.

Registered primary results:

| Method | Hard-negative triplet accuracy |
|---|---:|
| Raw CodeBERT cosine | `0.532458984375` |
| Raw CodeBERT Euclidean | `0.533314453125` |
| Predicted primitive executor | `0.732851562500` |
| Gap over best raw geometry | `0.199537109375` |

This result was independently recomputed in the original repository.

### Still unanswered

The stronger question remains unanswered:

> Does the primitive-coordinate approach materially outperform strong supervised metric-learning and directly trained pair models given equivalent training access?

The second attempt did not answer this because RELATE-E01 stopped before its primary test.

## Repository rule

The scientific question controls the project. Infrastructure may support an experiment, but it may not become the project itself.

The new repository will therefore begin with:

- one concise research statement;
- one migration inventory;
- a small set of extracted, understandable scientific modules;
- one finite next experiment;
- ordinary reproducibility protections against mistakes;
- an explicit stopping rule.

It will not initially include:

- generalized publication machinery;
- capability or authorization frameworks;
- tamper-proof ledger systems;
- agent workflows;
- broad plugin architectures;
- large historical artifact trees;
- inherited experiment numbering.

See [`docs/migration-inventory.md`](docs/migration-inventory.md) and [`docs/scientific-state.md`](docs/scientific-state.md).
