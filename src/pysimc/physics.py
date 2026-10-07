"""Reusable physics formulae (vectorised).  Energies MeV, cross sections ub/sr.

The form factors here are a *placeholder* (dipole).  Port the Fortran choices
(``physics_proton.f``: Galster / Bosted / deForest CC1 ...) behind the same
``form_factors`` signature when reproducing legacy numbers.
"""

from __future__ import annotations

import numpy as np

from .constants import ALPHA, FM2_TO_UB, HBARC, MP, MU_P


def mott_xsec(e_in, theta):
    """Point-like Mott cross section  alpha^2 cos^2(t/2) / (4 E^2 sin^4(t/2))  [ub/sr]."""
    s2 = np.sin(theta / 2.0) ** 2
    c2 = np.cos(theta / 2.0) ** 2
    return (ALPHA ** 2 * HBARC ** 2) * c2 / (4.0 * e_in ** 2 * s2 ** 2) * FM2_TO_UB


def dipole_form_factors(q2):
    """(GE, GM) of the proton, dipole model.  q2 in MeV^2."""
    gd = (1.0 + q2 / 0.71e6) ** -2
    return gd, MU_P * gd


def rosenbluth_xsec(e_in, theta, q2, form_factors=dipole_form_factors, mass=MP):
    """Elastic e-p dsigma/dOmega_e [ub/sr], recoil included."""
    ge, gm = form_factors(q2)
    tau = q2 / (4.0 * mass ** 2)
    eprime = e_in / (1.0 + (e_in / mass) * (1.0 - np.cos(theta)))
    tan2 = np.tan(theta / 2.0) ** 2
    return mott_xsec(e_in, theta) * (eprime / e_in) * ((ge ** 2 + tau * gm ** 2) / (1.0 + tau) + 2.0 * tau * gm ** 2 * tan2)
