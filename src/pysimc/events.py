"""Structure-of-arrays event container.

The Fortran code passes one event at a time through deeply nested structures.
Here a batch of N events is a dict of equal-length numpy arrays, so every
physics step is one vectorised call.  Column names are the shared vocabulary
between reactions, spectrometers, cuts and histograms.

Conventional columns (MeV, rad unless noted):
    weight                 cross section x phase space [ub] (per generated event)
    Ein, Eprime, theta_e   beam energy, scattered electron energy / angle
    nu, Q2, W, Em, Pm      kinematics (Q2 in MeV^2)
    e_delta, e_xptar, e_yptar, p_delta, p_xptar, p_yptar   arm coordinates
        (delta in %, slopes dimensionless)
    vx, vy, vz             vertex [cm]
"""

from __future__ import annotations

import numpy as np


class EventBatch:
    def __init__(self, n: int, cols: dict[str, np.ndarray] | None = None):
        self.n = int(n)
        self.cols: dict[str, np.ndarray] = {}
        for k, v in (cols or {}).items():
            self[k] = v

    def __getitem__(self, key: str) -> np.ndarray:
        try:
            return self.cols[key]
        except KeyError:
            raise KeyError(f"event column {key!r} not available; have {sorted(self.cols)}") from None

    def __setitem__(self, key: str, value) -> None:
        arr = np.asarray(value)
        if arr.shape != (self.n,):
            raise ValueError(f"column {key!r} has shape {arr.shape}, expected ({self.n},)")
        self.cols[key] = arr

    def __contains__(self, key: str) -> bool:
        return key in self.cols

    def __len__(self) -> int:
        return self.n

    def select(self, mask: np.ndarray) -> "EventBatch":
        mask = np.asarray(mask, dtype=bool)
        return EventBatch(int(mask.sum()), {k: v[mask] for k, v in self.cols.items()})
