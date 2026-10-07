import numpy as np
import pytest

from pysimc.constants import MP
from pysimc.engine import Simulation
from pysimc.kinematics import BEAM, slopes_to_direction


def test_two_body_kinematics_closes(base_cfg):
    """Energy and 3-momentum are conserved, W = M_p, Em = Pm = 0."""
    sim = Simulation(base_cfg)
    b = sim.reaction.generate(np.random.default_rng(3), 20_000)
    np.testing.assert_allclose(b["W"], MP, rtol=0, atol=1e-6)

    e_p = np.sqrt(b["p_mom"] ** 2 + MP ** 2)
    np.testing.assert_allclose(e_p, b["nu"] + MP, rtol=1e-12)

    u_e = slopes_to_direction(np.deg2rad(25.9), "right", b["e_xptar"], b["e_yptar"])
    u_p = slopes_to_direction(np.deg2rad(22.73), "left", b["p_xptar"], b["p_yptar"])
    balance = b["Eprime"][:, None] * u_e + b["p_mom"][:, None] * u_p - b["Ein"][:, None] * BEAM
    assert np.abs(balance).max() < 1e-8


def test_central_setting_is_near_elastic(base_cfg):
    """The legacy input was tuned so the central proton sits near delta ~ 0."""
    sim = Simulation(base_cfg)
    b = sim.reaction.generate(np.random.default_rng(4), 50_000)
    assert abs(np.median(b["e_delta"])) < 1.0
    assert abs(np.median(b["p_delta"])) < 2.0


def test_weights_positive_finite(base_cfg):
    b = Simulation(base_cfg).reaction.generate(np.random.default_rng(5), 10_000)
    assert np.all(np.isfinite(b["weight"])) and np.all(b["weight"] > 0)
