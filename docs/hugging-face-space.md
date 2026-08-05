# Hugging Face Space deployment

The Space is a live illustration of the preserved historical Option B mechanism.
It is not a new benchmark and does not alter RELATE-E01.

## 1. Export the frozen projection

From the RELATE checkout:

```powershell
python -m pip install -e ".[replay]"

python -m relate.export_space_artifact `
    --canonical-root C:\Projects\similarity_is_relative\artifacts\canonical\option-b `
    --cache C:\Projects\relate_new\.writer\option-b\cache\gpu-batch10-a.sqlite3 `
    --output C:\Projects\relate_new\space\assets\option-b-demo-projection.npz
```

The command reconstructs the canonical training and test embeddings from the
read-only SQLite cache, refits the three frozen `Ridge(alpha=1.0)` readouts, and
requires exact agreement with:

- every published coefficient hash;
- every published intercept hash;
- the complete 4,000-row canonical test prediction array.

A successful export ends with:

```text
SPACE_PROJECTION_EXPORTED_AND_VERIFIED
```

It creates:

```text
space/assets/option-b-demo-projection.npz
space/assets/option-b-demo-projection.json
```

## 2. Run locally

```powershell
python -m pip install -r .\space\requirements.txt
python .\space\app.py
```

The first request downloads the frozen CodeBERT revision. Submitted code is
parsed and embedded but never executed.

## 3. Publish to Hugging Face

Authenticate once:

```powershell
hf auth login
hf auth whoami
```

Create a Gradio Space, replacing `<namespace>` with the account reported by
`hf auth whoami`:

```powershell
hf repos create <namespace>/relate-demo `
    --type space `
    --space-sdk gradio `
    --exist-ok
```

Upload only the self-contained Space directory:

```powershell
hf upload <namespace>/relate-demo .\space `
    --type space `
    --commit-message "Publish verified RELATE Option B demo"
```

## What the demo shows

A user supplies one query function and two candidate functions. The Space
reports:

- CodeBERT cosine distance and ranking;
- Chebyshev distance in the frozen predicted relation space and ranking;
- Chebyshev distance in the measurable true AST relation and ranking;
- true and predicted values for cyclomatic complexity, maximum control depth,
  and distinct call sites.

The interface states clearly that arbitrary pasted functions are illustrative
and may be outside the original training distribution. The historical frozen
benchmark remains the evidence-bearing result.
