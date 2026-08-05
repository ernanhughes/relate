from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np

from relate import RelationProjection, extract_python_structure

MODEL_ID = "microsoft/codebert-base"
MODEL_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"
MAX_LENGTH = 256
MAX_SOURCE_CHARACTERS = 12_000
ARTIFACT_PATH = Path(__file__).parent / "assets" / "option-b-demo-projection.npz"

QUERY_EXAMPLE = '''def summarize_orders(orders):
    total = 0
    for order in orders:
        if order.is_valid():
            total += order.amount
    return round(total, 2)
'''

CANDIDATE_A_EXAMPLE = '''def total_valid_orders(orders):
    return round(sum(order.amount for order in orders if order.is_valid()), 2)
'''

CANDIDATE_B_EXAMPLE = '''def count_ready_jobs(jobs):
    ready = 0
    for job in jobs:
        if job.can_run():
            ready += job.weight
    return normalize(ready)
'''

SECOND_QUERY = '''def choose_path(node):
    if node.enabled:
        if node.children:
            return visit(node.children)
    return fallback(node)
'''

SECOND_A = '''def choose_path(node):
    return visit(node.children) if node.enabled and node.children else fallback(node)
'''

SECOND_B = '''def process_account(account):
    if account.active:
        if account.entries:
            return reconcile(account.entries)
    return archive(account)
'''


def _validate_source(name: str, source: str) -> str:
    value = source.strip()
    if not value:
        raise gr.Error(f"{name} is empty")
    if len(value) > MAX_SOURCE_CHARACTERS:
        raise gr.Error(
            f"{name} is too long; keep the example below {MAX_SOURCE_CHARACTERS:,} characters"
        )
    try:
        extract_python_structure(value)
    except (SyntaxError, ValueError) as exc:
        raise gr.Error(f"{name} must contain exactly one valid top-level Python function: {exc}")
    return value


@lru_cache(maxsize=1)
def _load_projection() -> RelationProjection:
    if not ARTIFACT_PATH.is_file():
        raise RuntimeError(
            "The verified projection artifact is missing. Export "
            "space/assets/option-b-demo-projection.npz before deploying the Space."
        )
    return RelationProjection.load(ARTIFACT_PATH)


@lru_cache(maxsize=1)
def _load_backend():
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.eval()
    return tokenizer, model, torch


def _embed(codes: list[str]) -> np.ndarray:
    tokenizer, model, torch = _load_backend()
    encoded = tokenizer(
        codes,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled.detach().cpu().numpy().astype(np.float32, copy=False)


def _cosine_distance(source: np.ndarray, targets: np.ndarray) -> np.ndarray:
    source64 = np.asarray(source, dtype=np.float64)
    targets64 = np.asarray(targets, dtype=np.float64)
    source_norm = np.linalg.norm(source64)
    target_norm = np.linalg.norm(targets64, axis=1)
    if source_norm == 0.0 or np.any(target_norm == 0.0):
        raise gr.Error("CodeBERT produced a zero-norm embedding")
    similarity = (targets64 @ source64) / (target_norm * source_norm)
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def _rank(values: np.ndarray) -> list[int]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return [int(value) for value in ranks]


def compare_relations(query: str, candidate_a: str, candidate_b: str):
    query = _validate_source("Query", query)
    candidate_a = _validate_source("Candidate A", candidate_a)
    candidate_b = _validate_source("Candidate B", candidate_b)

    projection = _load_projection()
    embeddings = _embed([query, candidate_a, candidate_b])
    predicted_scaled = np.asarray(projection.project(embeddings), dtype=np.float64)
    predicted_raw = (
        predicted_scaled * projection.relation_scale + projection.relation_median
    )
    true_raw = np.vstack(
        [
            extract_python_structure(source).as_array()
            for source in (query, candidate_a, candidate_b)
        ]
    )
    true_scaled = (
        true_raw - projection.relation_median
    ) / projection.relation_scale

    cosine = _cosine_distance(embeddings[0], embeddings[1:])
    predicted_relation = np.max(
        np.abs(predicted_scaled[1:] - predicted_scaled[0]), axis=1
    )
    true_relation = np.max(np.abs(true_scaled[1:] - true_scaled[0]), axis=1)

    cosine_ranks = _rank(cosine)
    predicted_ranks = _rank(predicted_relation)
    true_ranks = _rank(true_relation)

    ranking_rows = []
    for index, name in enumerate(("Candidate A", "Candidate B")):
        ranking_rows.append(
            [
                name,
                round(float(cosine[index]), 6),
                cosine_ranks[index],
                round(float(predicted_relation[index]), 6),
                predicted_ranks[index],
                round(float(true_relation[index]), 6),
                true_ranks[index],
            ]
        )

    coordinate_rows = []
    for row_index, name in enumerate(("Query", "Candidate A", "Candidate B")):
        coordinate_rows.append(
            [
                name,
                round(float(true_raw[row_index, 0]), 3),
                round(float(predicted_raw[row_index, 0]), 3),
                round(float(true_raw[row_index, 1]), 3),
                round(float(predicted_raw[row_index, 1]), 3),
                round(float(true_raw[row_index, 2]), 3),
                round(float(predicted_raw[row_index, 2]), 3),
            ]
        )

    cosine_winner = "A" if cosine[0] <= cosine[1] else "B"
    relate_winner = "A" if predicted_relation[0] <= predicted_relation[1] else "B"
    oracle_winner = "A" if true_relation[0] <= true_relation[1] else "B"
    agreement = "agrees" if relate_winner == oracle_winner else "does not agree"
    disagreement = (
        "Cosine and RELATE choose different candidates."
        if cosine_winner != relate_winner
        else "Cosine and RELATE choose the same candidate for this example."
    )

    summary = f"""
### Result

- **Cosine geometry chooses Candidate {cosine_winner}.**
- **The frozen RELATE projection chooses Candidate {relate_winner}.**
- **The true AST relation chooses Candidate {oracle_winner}.**
- The learned relation readout **{agreement}** with the measurable AST relation here.

{disagreement}

The relation distance is Chebyshev distance over robust-scaled predicted
**cyclomatic complexity**, **maximum control depth**, and **distinct call sites**.
Lower is closer.

> This live comparison is illustrative, not a new benchmark result. Arbitrary
> pasted functions may be outside the training distribution. The frozen
> historical benchmark result remains `0.7328515625` for the predicted executor,
> versus `0.532458984375` for cosine and `0.533314453125` for Euclidean distance.
"""
    return summary, ranking_rows, coordinate_rows


CSS = """
.gradio-container { max-width: 1180px !important; }
.hero { padding: 1.2rem 0 0.4rem 0; }
.metric-card { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px; }
footer { display: none !important; }
"""

with gr.Blocks(title="RELATE — Relation-aware embedding search") as demo:
    gr.Markdown(
        """
<div class="hero">

# RELATE: The embedding knew more than its geometry showed

Paste one query function and two candidates. The demo compares the candidates
using ordinary CodeBERT cosine distance and a frozen ridge projection into three
objective Python-structure coordinates.

</div>
"""
    )

    with gr.Row():
        gr.Markdown("**Cosine**\n\n`0.532458984375`", elem_classes="metric-card")
        gr.Markdown("**Raw Euclidean**\n\n`0.533314453125`", elem_classes="metric-card")
        gr.Markdown("**RELATE projection**\n\n`0.7328515625`", elem_classes="metric-card")
        gr.Markdown("**Gain over raw best**\n\n`+0.199537109375`", elem_classes="metric-card")

    gr.Markdown(
        """
The historical benchmark used 20,000 training functions, 4,000 test queries,
and 512,000 frozen hard-negative comparisons. This Space uses the same frozen
CodeBERT revision, tokenization, pooling, robust scaler, and three final ridge
readouts. Submitted code is parsed and embedded; it is **never executed**.
"""
    )

    with gr.Row():
        query_box = gr.Code(
            value=QUERY_EXAMPLE,
            language="python",
            label="Query function",
            lines=16,
        )
        candidate_a_box = gr.Code(
            value=CANDIDATE_A_EXAMPLE,
            language="python",
            label="Candidate A",
            lines=16,
        )
        candidate_b_box = gr.Code(
            value=CANDIDATE_B_EXAMPLE,
            language="python",
            label="Candidate B",
            lines=16,
        )

    compare_button = gr.Button("Compare cosine and RELATE", variant="primary")
    summary_output = gr.Markdown()

    gr.Markdown("## Candidate ranking")
    ranking_output = gr.Dataframe(
        headers=[
            "Candidate",
            "Cosine distance",
            "Cosine rank",
            "Predicted relation distance",
            "RELATE rank",
            "True AST relation distance",
            "Oracle rank",
        ],
        datatype=["str", "number", "number", "number", "number", "number", "number"],
        interactive=False,
        row_count=2,
        column_count=7,
    )

    gr.Markdown("## Structural coordinates")
    coordinate_output = gr.Dataframe(
        headers=[
            "Function",
            "True complexity",
            "Predicted complexity",
            "True control depth",
            "Predicted control depth",
            "True call sites",
            "Predicted call sites",
        ],
        datatype=["str", "number", "number", "number", "number", "number", "number"],
        interactive=False,
        row_count=3,
        column_count=7,
    )

    gr.Examples(
        examples=[
            [QUERY_EXAMPLE, CANDIDATE_A_EXAMPLE, CANDIDATE_B_EXAMPLE],
            [SECOND_QUERY, SECOND_A, SECOND_B],
        ],
        inputs=[query_box, candidate_a_box, candidate_b_box],
        label="Examples",
    )

    with gr.Accordion("Scientific boundary", open=False):
        gr.Markdown(
            """
This Space demonstrates the mechanism behind the preserved historical Option B
result. It does not reopen RELATE-E01, which remains `EXPERIMENT_INVALID`. It
does not claim that every relation is recoverable from every embedding, or that
the live examples constitute a new evaluation.
"""
        )

    compare_button.click(
        fn=compare_relations,
        inputs=[query_box, candidate_a_box, candidate_b_box],
        outputs=[summary_output, ranking_output, coordinate_output],
        api_name="compare_relations",
    )

if __name__ == "__main__":
    demo.launch(css=CSS)
