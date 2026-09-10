"""Evaluation cards: record what a representation achieved on a task."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EvaluationCard:
    task: str
    corpus: str
    space_hash: str
    metrics: dict = field(default_factory=dict)
    baselines: dict = field(default_factory=dict)
    per_relation: dict = field(default_factory=dict)
    manifest: str = ""

    def metric(self, name: str) -> float | None:
        value = self.metrics.get(name)
        return None if value is None else float(value)

    def beats_random(self, name: str) -> bool | None:
        if name not in self.metrics or name not in self.baselines:
            return None
        return float(self.metrics[name]) > float(self.baselines[name])
