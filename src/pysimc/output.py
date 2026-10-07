"""Reports and files.  Plain numpy/JSON/text so there are no heavy dependencies.

TODO: optional ROOT/HDF5/parquet writers (the Fortran code wrote HBOOK/ntuples);
keep them behind extras so the core stays dependency-light.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import SimConfig
from .constants import MP
from .results import Result
from .target import luminosity_per_ub


def summary(cfg: SimConfig, res: Result) -> dict:
    lumi = luminosity_per_ub(cfg.beam, cfg.target)
    return {
        "reaction": cfg.reaction.name,
        "n_tried": res.n_tried,
        "n_accepted": res.n_accepted,
        "acceptance": res.acceptance,
        "sigma_ub": res.sigma_ub,
        "sigma_err_ub": res.sigma_err_ub,
        "luminosity_per_ub": lumi,
        "expected_events": res.sigma_ub * lumi,
        "expected_events_err": res.sigma_err_ub * lumi,
        **res.meta,
    }


def format_report(cfg: SimConfig, res: Result) -> str:
    s = summary(cfg, res)
    e, p = cfg.electron_arm, cfg.hadron_arm
    lines = [
        "=" * 64,
        f" pysimc   reaction: {cfg.reaction.name}",
        "=" * 64,
        f" beam        {cfg.beam.energy:10.2f} MeV   charge {cfg.beam.charge_mC:g} mC",
        f" e arm       {e.spectrometer:>5}  P={e.momentum:9.2f} MeV/c  theta={e.theta_deg:6.2f} deg ({e.side})",
        f" hadron arm  {p.spectrometer:>5}  P={p.momentum:9.2f} MeV/c  theta={p.theta_deg:6.2f} deg ({p.side})",
        "-" * 64,
        f" trials          {s['n_tried']:>14,d}",
        f" accepted        {s['n_accepted']:>14,d}   ({100 * s['acceptance']:.3f} %)",
        f" sigma (in acc.) {s['sigma_ub']:>14.6e} +- {s['sigma_err_ub']:.2e} ub",
        f" luminosity      {s['luminosity_per_ub']:>14.6e} ub^-1",
        f" expected events {s['expected_events']:>14.6e} +- {s['expected_events_err']:.2e}",
        "-" * 64,
        f" seed {s['seed']}   workers {s['workers']}   wall {s['wall_time_s']:.2f} s"
        f"   ({s['n_tried'] / max(s['wall_time_s'], 1e-9):,.0f} trials/s)",
        "=" * 64,
    ]
    return "\n".join(lines)


def write_outputs(cfg: SimConfig, res: Result, directory: str | Path | None = None) -> Path:
    out = Path(directory or cfg.output.directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary(cfg, res), indent=2))
    (out / "report.txt").write_text(format_report(cfg, res) + "\n")
    (out / "config_used.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=list))
    if res.hists:
        arrays = {}
        for name, h in res.hists.items():
            arrays[f"{name}__edges"] = h.edges
            arrays[f"{name}__counts"] = h.counts
            arrays[f"{name}__errors"] = h.errors
        np.savez(out / "histograms.npz", **arrays)
    if cfg.output.ntuple:
        np.savez_compressed(out / "events.npz", **res.event_columns())
    return out
