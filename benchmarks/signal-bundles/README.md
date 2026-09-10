# Signal-bundle benchmark

Score alone struggles on a hard mix; the geometric bundle rescues it; a
generic external verifier adds nothing reliable.

## Ablation (balanced accuracy, train-tuned linear probe, test split)

| configuration | mirror | historic |
|---|---|---|
| score only | 0.677 | 0.76 |
| + margin (3A) | 0.854 | — |
| + density (3B) | 0.866 | — |
| + hubness (3B) | 0.877 | — |
| all-geometric | 0.906 | 0.90 |
| external alone (simulated) | 0.730 | — |
| stacked (geo + verdict feature) | 0.906 | 0.897 |

The probe is benchmark methodology, not runtime: bundles never classify,
fit, or decide. Every test bundle flows through production APIs -- 3A
observations for margins, a real 3C fit on train scores for the
calibration field, `Observatory.inspect_result` for composition. The
bundle carries the case margin; presentation sign is benchmark framing.

## External boundary

The verifier is simulated (`simulated-generic-verifier-v1`): right where
geometry is confident, coin-flip where it is hard -- the correlated
mistakes a generic second model actually makes. It is reported
separately and stacked the book's way (one more probe feature), so
redundancy reads as redundancy. No NLI or provider dependency exists
anywhere in `src/relate`.

## Completeness

Every 20th test bundle omits calibration (`available_signals` lacks the
field): partial but valid. `summary.json` records the full/partial
counts. Missing and measured-bad are never conflated.

## Running

```text
python benchmarks/signal-bundles/run.py          # write expected/
python benchmarks/signal-bundles/run.py --check  # verify byte-identical
```

Regression gates assert the shape (score in [0.60, 0.80], geometric gain
> 0.10, endpoint in [0.85, 0.95], stacked within 0.03 of geometric), not
sacred constants. Evidence categories stay labeled: HISTORIC lives in
`historic-reference.json`, everything else is mirror.
