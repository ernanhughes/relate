# Compression benchmark

Dimensionality is not a permission boundary. Preservation is.

## Sweep (PCA / random projection / prefix truncation × 64/32/16/8)

| config | retrieval | fine_ordering | threshold_transfer |
|---|---|---|---|
| pca-64 | PASS | PASS | FAIL |
| pca-32 | FAIL | PASS | FAIL |
| pca-16 | FAIL | PASS | FAIL |
| pca-8 | FAIL | FAIL | FAIL |
| random-* / prefix-* | FAIL | FAIL | FAIL |

PCA-32 is the honest middle: it fails retrieval policy while passing
fine-ordering policy. Same artifact, different scopes, different
verdicts -- knees live in (evidence, policy), never in the artifact.

## Task knees (PCA, declared policies in `policies.json`)

```text
retrieval           64 dims
fine_ordering       16 dims
threshold_transfer  no tested dimension passes
```

The useful statement is "this compression is usable for X under this
evidence", never "32 dimensions is enough."

## Geometry without faithfulness

PCA-8 keeps CKA 0.99 yet fails fine ordering (negation −0.10): global
geometry preserved, fine distinctions lost. Structured PCA beats the
random control on neighborhoods by 0.34 at matched width -- the task
tolerates loss, but structure still matters.

Prefix truncation is named neutrally (`training_support: unknown` in
provenance): slicing 32 coordinates is not Matryoshka training. PCA
explained variance and compression ratios travel as provenance
metadata; a 95%-variance cartridge can still fail a task, and here
transfer fails at every width.

## Running

```text
python benchmarks/compression/run.py          # write expected/
python benchmarks/compression/run.py --check  # verify byte-identical
```

Reference authority is the native source (`SOURCE_NATIVE` frame):
a cartridge preserves what the source did, or it does not. Fit corpus
(train correspondence hash) and evaluation correspondence are distinct
by construction; `knee.py` reads verdicts and judges nothing.
