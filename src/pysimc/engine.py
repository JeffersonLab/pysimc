"""The simulation core: one ``Simulation`` turns (chunk id, #trials, seed) into a ``Result``.

Pipeline for every batch (all vectorised):

    reaction.generate  ->  radiation.apply  ->  spectrometer acceptance
        ->  cuts  ->  accumulate (sums, histograms, optional event columns)

A *chunk* is the unit of parallel work and of random-number streams: chunk i
always uses the stream ``SeedSequence(seed, spawn_key=(i,))`` no matter which
process runs it, so results do not depend on the worker count.
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig
from .events import EventBatch
from .radiative import build_radiation
from .reactions import REACTIONS
from .results import Hist1D, Result
from .spectrometers import build_spectrometer


class Simulation:
    def __init__(self, cfg: SimConfig):
        cfg.validate()
        self.cfg = cfg
        self.reaction = REACTIONS.get(cfg.reaction.name)(cfg)
        self.radiation = build_radiation(cfg)
        self.spec = {"e": build_spectrometer(cfg.electron_arm),
                     "p": build_spectrometer(cfg.hadron_arm)}

    # -- one batch ---------------------------------------------------------------
    def process_batch(self, n: int, rng: np.random.Generator) -> EventBatch:
        """Generate n events and return only those that are accepted."""
        batch = self.reaction.generate(rng, n)
        batch = self.radiation.apply(batch, rng)
        keep = np.isfinite(batch["weight"]) & (batch["weight"] > 0)
        for arm in self.cfg.require_arms:
            keep &= self.spec[arm].accept(
                batch[f"{arm}_delta"], batch[f"{arm}_xptar"], batch[f"{arm}_yptar"],
                batch["vx"], batch["vy"], batch["vz"], rng)
        for var, (lo, hi) in self.cfg.cuts.items():
            keep &= (batch[var] >= lo) & (batch[var] <= hi)
        return batch.select(keep)

    # -- one chunk ---------------------------------------------------------------
    def run_chunk(self, chunk_id: int, n_trials: int, seed: int) -> Result:
        rng = np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(chunk_id,)))
        out = self.cfg.output
        res = Result(hists={h.name or h.var: Hist1D.from_config(h) for h in out.histograms})
        remaining = n_trials
        while remaining > 0:
            n = min(self.cfg.run.batch_size, remaining)
            remaining -= n
            acc = self.process_batch(n, rng)
            res.n_tried += n
            res.n_accepted += len(acc)
            if len(acc) == 0:
                continue
            w = acc["weight"]
            res.sum_w += float(w.sum())
            res.sum_w2 += float(np.square(w).sum())
            for h in res.hists.values():
                h.fill(acc[h.var], w)
            if out.ntuple:
                names = out.ntuple_columns or sorted(acc.cols)
                for name in names:
                    res.columns.setdefault(name, []).append(acc[name].copy())
        return res
