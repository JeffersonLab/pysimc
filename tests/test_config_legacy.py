import tomllib
import warnings
from pathlib import Path

import numpy as np
import pytest

from pysimc.config import SimConfig
from pysimc.legacy import load_legacy_inp, to_toml
from pysimc.spectrometers import PolynomialMatrix

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown config key"):
        SimConfig.from_dict({"beam": {"enrgy": 5000.0}})


def test_validation():
    cfg = SimConfig.from_dict({"beam": {"energy": 0.0}})
    with pytest.raises(ValueError):
        cfg.validate()


def test_example_toml_loads():
    cfg = SimConfig.from_file(EXAMPLES / "hydrogen_elastic.toml")
    cfg.validate()
    assert cfg.hadron_arm.delta == (-15.0, 15.0)
    assert cfg.output.histograms[0].var == "Q2"


def test_legacy_conversion_roundtrip():
    with pytest.warns(UserWarning, match="not mapped yet"):
        d = load_legacy_inp(EXAMPLES / "legacy" / "eep_hydrogen_q8.inp")
    assert d["reaction"]["name"] == "hydrogen_elastic"
    assert d["electron_arm"]["spectrometer"] == "hms" and d["electron_arm"]["side"] == "right"
    assert d["hadron_arm"]["spectrometer"] == "sos" and d["hadron_arm"]["side"] == "left"
    assert d["run"] == {"mode": "accepted", "n_accepted": 5000}
    assert d["beam"]["xwid"] == pytest.approx(0.008868)        # came from the extra_dbase_file
    cfg = SimConfig.from_dict(tomllib.loads(to_toml(d)))
    cfg.validate()
    assert cfg.electron_arm.momentum == 4531.0


def test_polynomial_matrix():
    # f(x, y) = 2 + 3x + x*y^2
    pm = PolynomialMatrix([[0, 0], [1, 0], [1, 2]], [2.0, 3.0, 1.0], chunk=3)
    x = np.random.default_rng(0).normal(size=(10, 2))
    np.testing.assert_allclose(pm.evaluate(x), 2 + 3 * x[:, 0] + x[:, 0] * x[:, 1] ** 2)


def test_stub_reaction_raises(base_cfg):
    import dataclasses
    from pysimc.engine import Simulation
    cfg = dataclasses.replace(base_cfg, reaction=dataclasses.replace(base_cfg.reaction, name="pion"))
    with pytest.raises(NotImplementedError, match="physics_pion.f"):
        Simulation(cfg)


def test_unported_radiation_fails_at_startup(base_cfg):
    import dataclasses
    from pysimc.engine import Simulation
    cfg = dataclasses.replace(base_cfg, radiation=dataclasses.replace(base_cfg.radiation, model="mo_tsai"))
    with pytest.raises(NotImplementedError, match="radc.f"):
        Simulation(cfg)
