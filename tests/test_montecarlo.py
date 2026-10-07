"""Statistical correctness of the Monte Carlo machinery against direct quadrature."""
import dataclasses

import numpy as np
import pytest

from pysimc.constants import MP
from pysimc.kinematics import slopes_to_direction, solid_angle_jacobian, BEAM
from pysimc.physics import rosenbluth_xsec
from pysimc.runner import run
from conftest import replace_run


def quadrature_sigma(cfg, n=600):
    """Integrate dsigma/dOmega over the electron slope box with a midpoint grid."""
    e = cfg.electron_arm
    xs = np.linspace(*(np.array(e.xptar_mrad) * 1e-3), n + 1)
    ys = np.linspace(*(np.array(e.yptar_mrad) * 1e-3), n + 1)
    xm, ym = 0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1])
    X, Y = np.meshgrid(xm, ym, indexing="ij")
    u = slopes_to_direction(np.deg2rad(e.theta_deg), e.side, X.ravel(), Y.ravel())
    cos_t = u @ BEAM
    ein = cfg.beam.energy
    th = np.arccos(cos_t)
    ep = ein / (1 + ein / MP * (1 - cos_t))
    q2 = 2 * ein * ep * (1 - cos_t)
    from pysimc.physics import dipole_form_factors
    sig = rosenbluth_xsec(ein, th, q2, dipole_form_factors)
    jac = solid_angle_jacobian(X.ravel(), Y.ravel())
    return float((sig * jac).sum() * (xs[1] - xs[0]) * (ys[1] - ys[0]))


def singles_cfg(cfg):
    """Electron-arm singles, no beam spread/raster, delta window wide open."""
    return dataclasses.replace(
        cfg,
        beam=dataclasses.replace(cfg.beam, energy_spread_pct=0.0, xwid=0.0, ywid=0.0, raster_pattern=0),
        electron_arm=dataclasses.replace(cfg.electron_arm, delta=(-99.0, 100.0)),
        require_arms=["e"],
        run=dataclasses.replace(cfg.run, n_trials=200_000),
    )


def test_mc_matches_quadrature(base_cfg):
    cfg = singles_cfg(base_cfg)
    res = run(cfg, workers=1)
    exact = quadrature_sigma(cfg)
    assert res.n_accepted == res.n_tried                     # nothing rejected
    assert abs(res.sigma_ub - exact) < 4 * res.sigma_err_ub
    assert res.sigma_err_ub / exact < 0.01                   # error estimate is sane


def test_coincidence_acceptance_reduces_sigma(base_cfg):
    singles = run(singles_cfg(base_cfg), workers=1)
    coinc = run(dataclasses.replace(singles_cfg(base_cfg), require_arms=["e", "p"]), workers=1)
    assert coinc.sigma_ub < singles.sigma_ub
    assert 0 < coinc.n_accepted < coinc.n_tried


def test_histogram_integral_equals_sigma(base_cfg):
    cfg = dataclasses.replace(
        base_cfg, output=dataclasses.replace(base_cfg.output, histograms=[
            dataclasses.replace(base_cfg.output.histograms[0], range=(0.0, 1e8), bins=50)]))
    res = run(cfg, workers=1)
    h = res.hists["Q2"]
    assert h.counts.sum() / res.n_tried == pytest.approx(res.sigma_ub, rel=1e-9)


def test_accepted_mode_stops(base_cfg):
    cfg = replace_run(base_cfg, mode="accepted", n_accepted=5000, chunk_trials=10_000)
    res = run(cfg, workers=1)
    assert res.n_accepted >= 5000
    assert res.n_tried < 16 * 10_000 * 3        # a few waves at most
