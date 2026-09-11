"""Task knees: smallest width passing a scope. Benchmark analysis only.

A knee is a property of (evidence, declared policy), never of an
artifact: the same PCA-32 passes retrieval policy and fails transfer
policy. This helper reads verdicts; it never judges vectors.
"""


def find_smallest_passing_dimension(results: list[tuple[int, bool]]) -> int | None:
    """Smallest width with a PASS, or None when no tested width passes."""
    passing = sorted(width for width, passed in results if passed)
    return passing[0] if passing else None
