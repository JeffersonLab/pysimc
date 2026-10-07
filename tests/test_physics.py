import numpy as np
import pytest

from pysimc.config import BeamConfig, TargetConfig
from pysimc.physics import mott_xsec
from pysimc.target import luminosity_per_ub


def test_mott_known_value():
    # alpha^2 (hbar c)^2 cos^2(30) / (4 E^2 sin^4(30)) at E=1 GeV, 60 deg
    # = 6.2e-6 fm^2/sr = 0.062 ub/sr
    assert mott_xsec(1000.0, np.deg2rad(60.0)) == pytest.approx(0.0622, rel=5e-3)


def test_luminosity_matches_fortran_constant():
    beam = BeamConfig(charge_mC=1.0)
    tgt = TargetConfig(mass_amu=1.007276, thickness_mg_cm2=295.172, abundancy_pct=100.0)
    fortran = 1.0 / (tgt.mass_amu / 3.75914e6 / 1.0 / (tgt.thickness_mg_cm2))
    assert luminosity_per_ub(beam, tgt) == pytest.approx(fortran, rel=2e-3)


def test_luminosity_scales_with_tilt():
    beam = BeamConfig(charge_mC=2.0)
    flat = luminosity_per_ub(beam, TargetConfig(angle_deg=0.0))
    tilt = luminosity_per_ub(beam, TargetConfig(angle_deg=60.0))
    assert tilt == pytest.approx(2 * flat)
