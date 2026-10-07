"""Mergeable accumulators.

Everything a worker returns is an *additive* quantity (sums, histogram
contents), so merging chunks is just ``+``.  Merging in chunk order makes
results bit-for-bit independent of the number of worker processes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import HistogramConfig


@dataclass
class Hist1D:
    name: str
    var: str
    edges: np.ndarray
    counts: np.ndarray        # sum of weights
    sumw2: np.ndarray         # sum of weights^2 (-> statistical error)
    n_entries: int = 0

    @classmethod
    def from_config(cls, hc: HistogramConfig) -> "Hist1D":
        edges = np.linspace(hc.range[0], hc.range[1], hc.bins + 1)
        return cls(hc.name or hc.var, hc.var, edges, np.zeros(hc.bins), np.zeros(hc.bins))

    def fill(self, values: np.ndarray, weights: np.ndarray) -> None:
        c, _ = np.histogram(values, bins=self.edges, weights=weights)
        w2, _ = np.histogram(values, bins=self.edges, weights=weights ** 2)
        self.counts += c
        self.sumw2 += w2
        self.n_entries += int(np.count_nonzero((values >= self.edges[0]) & (values < self.edges[-1])))

    def merge(self, other: "Hist1D") -> None:
        if not np.array_equal(self.edges, other.edges):
            raise ValueError(f"cannot merge histogram {self.name!r}: binning differs")
        self.counts += other.counts
        self.sumw2 += other.sumw2
        self.n_entries += other.n_entries

    @property
    def errors(self) -> np.ndarray:
        return np.sqrt(self.sumw2)


@dataclass
class Result:
    n_tried: int = 0
    n_accepted: int = 0
    sum_w: float = 0.0
    sum_w2: float = 0.0
    hists: dict[str, Hist1D] = field(default_factory=dict)
    columns: dict[str, list[np.ndarray]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def merge(self, other: "Result") -> None:
        self.n_tried += other.n_tried
        self.n_accepted += other.n_accepted
        self.sum_w += other.sum_w
        self.sum_w2 += other.sum_w2
        for name, h in other.hists.items():
            if name in self.hists:
                self.hists[name].merge(h)
            else:
                self.hists[name] = h
        for name, chunks in other.columns.items():
            self.columns.setdefault(name, []).extend(chunks)

    # -- derived ------------------------------------------------------------
    @property
    def sigma_ub(self) -> float:
        """Integrated cross section inside the acceptance [ub]."""
        return self.sum_w / self.n_tried if self.n_tried else 0.0

    @property
    def sigma_err_ub(self) -> float:
        n = self.n_tried
        if n < 2:
            return float("nan")
        mean = self.sum_w / n
        var = max(self.sum_w2 / n - mean * mean, 0.0)
        return math.sqrt(var / n)

    @property
    def acceptance(self) -> float:
        return self.n_accepted / self.n_tried if self.n_tried else 0.0

    def event_columns(self) -> dict[str, np.ndarray]:
        return {k: np.concatenate(v) for k, v in self.columns.items() if v}
