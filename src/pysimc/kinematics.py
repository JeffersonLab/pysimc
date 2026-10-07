"""Vectorised geometry between spectrometer slopes and lab directions.

Lab frame (same as SIMC):  z = beam direction, x = down, y = beam-left.
Spectrometer ray at slopes (xptar, yptar) around a central ray at angle theta0:

    direction  ~  z_s + xptar * x_s + yptar * y_s        (normalised)

with x_s = lab x (out of plane) and y_s the in-plane transverse axis.  This
reproduces ``physics_angles`` in the Fortran ``event.f``:
    cos(theta) = (cos(theta0) - yptar*sin(theta0)*sin(phi0)) / sqrt(1+x^2+y^2)
where sin(phi0) = +1 for a spectrometer on the left, -1 on the right.

NB: xptar is OUT of plane and yptar IN plane (the opposite of what one might
guess from the names); an earlier prototype got this backwards.
"""

from __future__ import annotations

import numpy as np

from .config import ArmConfig

BEAM = np.array([0.0, 0.0, 1.0])


def _side_sign(side: str) -> float:
    return 1.0 if side == "left" else -1.0     # sin(phi0)


def spectrometer_basis(theta0: float, side: str):
    """Return (x_s, y_s, z_s) lab-frame unit vectors; theta0 in radians."""
    s = _side_sign(side)
    c, sn = np.cos(theta0), np.sin(theta0)
    x_s = np.array([1.0, 0.0, 0.0])
    y_s = np.array([0.0, c, -sn * s])
    z_s = np.array([0.0, sn * s, c])
    return x_s, y_s, z_s


def slopes_to_direction(theta0: float, side: str, xptar, yptar) -> np.ndarray:
    """(n,) slopes -> (n,3) unit vectors in the lab frame."""
    x_s, y_s, z_s = spectrometer_basis(theta0, side)
    xptar = np.asarray(xptar, dtype=float)
    yptar = np.asarray(yptar, dtype=float)
    v = z_s + xptar[:, None] * x_s + yptar[:, None] * y_s
    return v / np.linalg.norm(v, axis=1)[:, None]


def direction_to_slopes(theta0: float, side: str, u: np.ndarray):
    """(n,3) lab unit vectors -> (xptar, yptar) in the spectrometer frame."""
    x_s, y_s, z_s = spectrometer_basis(theta0, side)
    dz = u @ z_s
    with np.errstate(divide="ignore", invalid="ignore"):
        return (u @ x_s) / dz, (u @ y_s) / dz


def solid_angle_jacobian(xptar, yptar):
    """dOmega = dxptar dyptar / (1 + x^2 + y^2)^(3/2)."""
    return (1.0 + np.square(xptar) + np.square(yptar)) ** -1.5


def arm_central_theta(arm: ArmConfig) -> float:
    return np.deg2rad(abs(arm.theta_deg))
