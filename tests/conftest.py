import dataclasses

import pytest

from pysimc.config import SimConfig


@pytest.fixture
def base_cfg() -> SimConfig:
    """Small H(e,e'p) elastic setup (HMS/SOS-like, 8.8 GeV)."""
    return SimConfig.from_dict({
        "run": {"mode": "trials", "n_trials": 120_000, "chunk_trials": 20_000,
                "batch_size": 7_000, "seed": 7},
        "beam": {"energy": 8800.0, "energy_spread_pct": 0.05, "charge_mC": 1.0,
                 "xwid": 0.009, "ywid": 0.004, "raster_pattern": 3,
                 "raster_1": 0.1, "raster_2": 0.1},
        "target": {"mass_amu": 1.007276, "rho": 0.07332, "thickness_mg_cm2": 295.172},
        "electron_arm": {"spectrometer": "hms", "side": "right", "momentum": 4531.0,
                         "theta_deg": 25.9, "delta": [-10, 10],
                         "xptar_mrad": [-100, 100], "yptar_mrad": [-50, 50]},
        "hadron_arm": {"spectrometer": "sos", "side": "left", "momentum": 5122.0,
                       "theta_deg": 22.73, "delta": [-15, 15],
                       "xptar_mrad": [-50, 50], "yptar_mrad": [-90, 90]},
        "reaction": {"name": "hydrogen_elastic"},
        "output": {"histograms": [{"name": "Q2", "var": "Q2", "bins": 20,
                                   "range": [6e6, 10e6]}]},
    })


def replace_run(cfg, **kw):
    return dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, **kw))
