"""Parallel driver.

Work is cut into fixed-size chunks (``run.chunk_trials``).  Chunks are farmed
out to a process pool (the GIL makes threads useless for numpy-light Python
code, and numpy releases the GIL only briefly) and merged **in chunk order**.
Because every chunk has its own RNG stream and the merge order is fixed, a run
gives identical numbers for ``workers=1`` and ``workers=N``.

``mode = "accepted"`` (legacy ``ngen > 0``) runs chunks in waves of
``WAVE`` chunks until enough events are accepted; the wave size is a
constant, not a function of the worker count, to keep results reproducible.
"""

from __future__ import annotations

import dataclasses
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import numpy as np

from .config import SimConfig
from .engine import Simulation
from .results import Result

WAVE = 16
_SIM: Simulation | None = None


def _init_worker(cfg: SimConfig) -> None:
    global _SIM
    _SIM = Simulation(cfg)


def _work(task: tuple[int, int, int]) -> Result:
    chunk_id, n_trials, seed = task
    assert _SIM is not None
    return _SIM.run_chunk(chunk_id, n_trials, seed)


def run(cfg: SimConfig, workers: int | None = None, seed: int | None = None,
        progress: Callable[[Result], None] | None = None) -> Result:
    """Run a full simulation.  ``workers=None`` uses all CPUs, ``1`` stays in-process."""
    cfg.validate()
    if seed is not None:
        cfg = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, seed=seed))
    if cfg.run.seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 63))
        cfg = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, seed=seed))
    seed = cfg.run.seed
    workers = workers or os.cpu_count() or 1

    total = Result()
    t0 = time.perf_counter()
    pool = None
    sim = Simulation(cfg)       # build in the parent too: bad configs / unported physics fail here,
    try:                        # with a readable error, instead of inside a worker
        if workers > 1:
            pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(cfg,))
            run_tasks = lambda tasks: list(pool.map(_work, tasks))     # ordered
        else:
            run_tasks = lambda tasks: [sim.run_chunk(c, n, s) for c, n, s in tasks]

        for wave in _waves(cfg, seed):
            for res in run_tasks(wave):
                total.merge(res)
            if progress:
                progress(total)
            if _done(cfg, total):
                break
    finally:
        if pool is not None:
            pool.shutdown()

    total.meta.update(seed=seed, workers=workers, wall_time_s=time.perf_counter() - t0,
                      chunk_trials=cfg.run.chunk_trials)
    return total


def _waves(cfg: SimConfig, seed: int):
    """Yield lists of (chunk_id, n_trials, seed)."""
    ct = cfg.run.chunk_trials
    if cfg.run.mode == "trials":
        n = cfg.run.n_trials
        tasks = [(i, min(ct, n - i * ct), seed) for i in range((n + ct - 1) // ct)]
        for lo in range(0, len(tasks), WAVE):
            yield tasks[lo:lo + WAVE]
    else:
        i = 0
        while True:
            yield [(i + k, ct, seed) for k in range(WAVE)]
            i += WAVE


def _done(cfg: SimConfig, total: Result) -> bool:
    if cfg.run.mode == "accepted":
        return total.n_accepted >= cfg.run.n_accepted or total.n_tried >= cfg.run.max_trials
    return False
