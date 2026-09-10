"""Geometry diagnostics: the shape of a space, never a verdict on it.

Answers how concentrated a space is, how anisotropic it is, how many
effective directions it uses, and what random-pair similarity looks like.
It never answers whether an embedding model is good: geometry diagnoses
geometry first, and similarity magnitude is space-relative.

No RELATE relation vocabulary, no bridge logic, no plotting/projection
artifacts (PCA coordinates, UMAP maps) -- those are outputs derived from
geometry, not geometry itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from relate.model import RelateError

QUANTILE_KEYS = ("p5", "p25", "p50", "p75", "p95")
QUANTILE_LEVELS = (5.0, 25.0, 50.0, 75.0, 95.0)


@dataclass(frozen=True, slots=True)
class PairSamplingSpec:
    """Deterministic random-pair sampling policy.

    A census (every pair) is used when the pair count fits ``max_pairs``;
    otherwise uniform sampling without replacement under ``seed``. The
    realized ``n_pairs`` and ``census`` flag travel in the report so two
    cosine means measured under different sampling are never compared
    silently.
    """

    mode: str = "uniform_without_replacement"
    max_pairs: int = 100_000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.mode != "uniform_without_replacement":
            raise RelateError(f"unknown sampling mode: {self.mode}")
        if (
            not isinstance(self.max_pairs, int)
            or isinstance(self.max_pairs, bool)
            or self.max_pairs <= 0
        ):
            raise RelateError("max_pairs must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise RelateError("seed must be an integer")


@dataclass(frozen=True, slots=True)
class IntrinsicDimensionEstimate:
    """An estimator-dependent guess, never a property of the space itself."""

    method: str
    estimate: float
    n_samples: int
    parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.method:
            raise RelateError("method must be non-empty")
        if not math.isfinite(self.estimate) or self.estimate <= 0.0:
            raise RelateError("estimate must be finite and positive")


@dataclass(frozen=True, slots=True)
class GeometryReport:
    n_vectors: int
    dimensions: int
    norm_mean: float
    norm_std: float
    cosine_mean: float
    cosine_std: float
    cosine_quantiles: dict = field(default_factory=dict)
    effective_rank: float | None = None
    participation_ratio: float | None = None
    intrinsic_dimension: IntrinsicDimensionEstimate | None = None
    sampling: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def _matrix(name: str, value: npt.ArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise RelateError(f"{name} must be a two-dimensional matrix")
    if array.shape[0] < 2:
        raise RelateError(f"{name} needs at least two vectors")
    if not np.isfinite(array).all():
        raise RelateError(f"{name} must contain only finite values")
    return array


def _norms(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise RelateError("cosine geometry is undefined for zero-norm vectors")
    return norms


def sample_pairs(n: int, spec: PairSamplingSpec) -> tuple[np.ndarray, bool]:
    """Return (flat upper-triangle indices, census flag) deterministically."""
    total = n * (n - 1) // 2
    if total <= spec.max_pairs:
        return np.arange(total), True
    rng = np.random.default_rng(spec.seed)
    chosen = np.sort(rng.choice(total, size=spec.max_pairs, replace=False))
    return chosen, False


def covariance_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of the centered covariance (NumPy only)."""
    centered = matrix - matrix.mean(axis=0)
    covariance = (centered.T @ centered) / max(1, matrix.shape[0] - 1)
    values = np.linalg.eigvalsh(covariance)[::-1]
    return np.clip(values, 0.0, None)


def _effective_rank_from_eigenvalues(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0.0:
        raise RelateError("effective rank is undefined for a constant space")
    positive = values[values > 0.0] / total
    return float(np.exp(-np.sum(positive * np.log(positive))))


def _participation_ratio_from_eigenvalues(values: np.ndarray) -> float:
    denominator = float(np.sum(values * values))
    if denominator <= 0.0:
        raise RelateError("participation ratio is undefined for a constant space")
    return float(float(values.sum()) ** 2 / denominator)


def effective_rank(matrix: npt.ArrayLike) -> float:
    """Exp-entropy of the spectral distribution (Roy & Vetterli)."""
    return _effective_rank_from_eigenvalues(
        covariance_eigenvalues(_matrix("vectors", matrix))
    )


def participation_ratio(matrix: npt.ArrayLike) -> float:
    """(sum eigenvalues)^2 / sum eigenvalues^2: soft count of directions."""
    return _participation_ratio_from_eigenvalues(
        covariance_eigenvalues(_matrix("vectors", matrix))
    )


def twonn_estimate(
    matrix: npt.ArrayLike, *, parameters: dict | None = None
) -> IntrinsicDimensionEstimate:
    """TwoNN intrinsic-dimension estimate (Facco et al. 2017, NumPy only).

    From the ratios mu = r2/r1 of second-to-first neighbor distances: under
    local uniformity, log(mu) is exponential with rate equal to the
    intrinsic dimension. O(n^2) memory; suited to thousands of vectors.
    """
    array = _matrix("vectors", matrix)
    n = array.shape[0]
    first = np.empty(n)
    second = np.empty(n)
    for row in range(n):
        distances = np.linalg.norm(array - array[row], axis=1)
        distances[row] = np.inf
        smallest = np.argpartition(distances, 2)[:2]
        first[row], second[row] = distances[smallest[0]], distances[smallest[1]]
        if first[row] > second[row]:
            first[row], second[row] = second[row], first[row]
    usable = (first > 0.0) & np.isfinite(second)
    ratios = second[usable] / first[usable]
    ratios = ratios[np.isfinite(ratios) & (ratios > 1.0)]
    if ratios.size < 10:
        raise RelateError("too few usable neighbor ratios for a TwoNN estimate")
    logs = np.log(ratios)
    # Exponential MLE: rate = 1 / mean(log mu).
    estimate = float(1.0 / logs.mean())
    return IntrinsicDimensionEstimate(
        method="twonn",
        estimate=estimate,
        n_samples=int(array.shape[0]),
        parameters=dict(parameters or {}),
    )


def linear_cka(first: npt.ArrayLike, second: npt.ArrayLike) -> float:
    """Linear CKA between two same-row-count spaces (Kornblith et al.).

    High CKA indicates related representational structure under the chosen
    sample. It does not authorize mixing vectors or transferring thresholds.
    """
    x = _matrix("first", first)
    y = _matrix("second", second)
    if x.shape[0] != y.shape[0]:
        raise RelateError("CKA needs the same row count in both spaces")

    def centered_gram(matrix: np.ndarray) -> np.ndarray:
        gram = matrix @ matrix.T
        return (
            gram
            - gram.mean(axis=0, keepdims=True)
            - gram.mean(axis=1, keepdims=True)
            + gram.mean()
        )

    k, ell = centered_gram(x), centered_gram(y)
    denominator = float(np.sum(k * k) * np.sum(ell * ell))
    if denominator <= 0.0:
        raise RelateError("CKA is undefined for constant spaces")
    return float(np.sum(k * ell) / math.sqrt(denominator))


def describe_geometry(
    vectors: npt.ArrayLike,
    *,
    sampling: PairSamplingSpec | None = None,
    with_spectrum: bool = True,
    with_twonn: bool = True,
    metadata: dict | None = None,
) -> GeometryReport:
    """Measure shape: concentration, anisotropy, effective directions."""
    array = _matrix("vectors", vectors)
    spec = sampling or PairSamplingSpec()
    norms = np.linalg.norm(array, axis=1)
    unit = array / _norms(array)[:, None]

    chosen, census = sample_pairs(array.shape[0], spec)
    rows, cols = np.triu_indices(array.shape[0], k=1)
    similarities = (unit[rows[chosen]] * unit[cols[chosen]]).sum(axis=1)
    quantiles = np.quantile(np.asarray(similarities), np.array(QUANTILE_LEVELS) / 100.0)

    eigenvalues = covariance_eigenvalues(array) if with_spectrum else None
    intrinsic = twonn_estimate(array) if with_twonn else None
    report_metadata = dict(metadata or {})
    return GeometryReport(
        n_vectors=int(array.shape[0]),
        dimensions=int(array.shape[1]),
        norm_mean=float(norms.mean()),
        norm_std=float(norms.std()),
        cosine_mean=float(similarities.mean()),
        cosine_std=float(similarities.std()),
        cosine_quantiles={
            key: float(value) for key, value in zip(QUANTILE_KEYS, quantiles)
        },
        effective_rank=(
            _effective_rank_from_eigenvalues(eigenvalues)
            if eigenvalues is not None
            else None
        ),
        participation_ratio=(
            _participation_ratio_from_eigenvalues(eigenvalues)
            if eigenvalues is not None
            else None
        ),
        intrinsic_dimension=intrinsic,
        sampling={
            "mode": spec.mode,
            "max_pairs": spec.max_pairs,
            "seed": spec.seed,
            "n_pairs": int(chosen.size),
            "census": bool(census),
        },
        metadata=report_metadata,
    )
