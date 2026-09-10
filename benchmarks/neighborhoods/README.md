# Neighborhood benchmark

Recovery is not fidelity. Three conditions over paired synthetic spaces
(`paired_spaces.py`: shared nuisance skeleton, signal blended 0.7/0.3):

| condition | 10-NN overlap | counterpart top-1 | MRR |
|---|---|---|---|
| identity (X -> X) | 1.000 | 1.000 | 1.000 |
| translated (X -> Y) | 0.626 | 1.000 | 1.000 |
| noise (X -> random) | 0.005 | 0.001 | 0.004 |

The translated row is the book's signature split, held in ONE report
(`space-comparison-xy.json`):

```text
CKA                        0.927
cosine-matrix correlation  0.927
counterpart recall@1       1.000
10-NN overlap              0.626
relation readout (X-fit)   0.725 -> 0.585  (delta -0.14 via 3A evaluator)
```

Coarse geometry looks strong while the fine distinction fails -- the same
structure as Wave 3's paraphrase-vs-negation inversion and Wave 6's
recovery-vs-structure split. Counterpart recovery and neighborhood
preservation are separate reports precisely so this contradiction survives.

`SpaceComparisonReport` carries the four measurements as evidence. It has
no `usable_for`: measurement, then evidence, then policy -- never
measurement = policy.

## Running

```text
python benchmarks/neighborhoods/run.py          # write expected/
python benchmarks/neighborhoods/run.py --check  # verify byte-identical
```

The signature split is asserted in `run.py` (CKA > 0.85, counterpart > 0.99,
overlap > 0.5, relation delta < -0.1), so regressions fail loudly. The 3A
hard-negative evaluator (`evaluate_hard_negatives`, `compare_reports`) is
reused unchanged -- bridges will later orchestrate these same functions.
