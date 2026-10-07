import numpy as np
import pytest

from pysimc.kinematics import (BEAM, direction_to_slopes, slopes_to_direction,
                               solid_angle_jacobian, spectrometer_basis)


@pytest.mark.parametrize("side", ["left", "right"])
def test_basis_is_orthonormal(side):
    x, y, z = spectrometer_basis(np.deg2rad(31.0), side)
    for a in (x, y, z):
        assert np.linalg.norm(a) == pytest.approx(1.0)
    assert x @ y == pytest.approx(0) and x @ z == pytest.approx(0) and y @ z == pytest.approx(0)


@pytest.mark.parametrize("side", ["left", "right"])
def test_slopes_roundtrip(side):
    rng = np.random.default_rng(1)
    xp, yp = rng.uniform(-0.1, 0.1, 500), rng.uniform(-0.1, 0.1, 500)
    u = slopes_to_direction(np.deg2rad(40), side, xp, yp)
    xr, yr = direction_to_slopes(np.deg2rad(40), side, u)
    np.testing.assert_allclose(xr, xp, atol=1e-12)
    np.testing.assert_allclose(yr, yp, atol=1e-12)


@pytest.mark.parametrize("side,s", [("left", 1.0), ("right", -1.0)])
def test_matches_fortran_physics_angles(side, s):
    """cos(theta) = (cos th0 - yptar sin th0 sin phi0)/sqrt(1+x^2+y^2)  (event.f)."""
    th0 = np.deg2rad(25.0)
    xp, yp = np.array([0.02, -0.03]), np.array([0.01, 0.04])
    u = slopes_to_direction(th0, side, xp, yp)
    expect = (np.cos(th0) - yp * np.sin(th0) * s) / np.sqrt(1 + xp**2 + yp**2)
    np.testing.assert_allclose(u @ BEAM, expect, atol=1e-12)


def test_sides_are_on_opposite_sides_of_beam():
    ul = slopes_to_direction(np.deg2rad(30), "left", [0.0], [0.0])[0]
    ur = slopes_to_direction(np.deg2rad(30), "right", [0.0], [0.0])[0]
    assert ul[1] > 0 > ur[1]


def test_jacobian_small_angle():
    assert solid_angle_jacobian(0.0, 0.0) == 1.0
    assert solid_angle_jacobian(0.1, 0.0) == pytest.approx(1.01 ** -1.5)
