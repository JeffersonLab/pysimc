"""Beam, target geometry and luminosity.

TODO (port from Fortran ``target.f`` / ``enerloss_new.f``):
  * energy loss + Coulomb acceleration (``EnergyLoss`` interface below)
  * radiation lengths / effective thickness for radiative corrections
  * target can geometry / endcap windows (``targ%can``)
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from .config import BeamConfig, TargetConfig
from .constants import E_CHARGE_C, N_AVOGADRO


def luminosity_per_ub(beam: BeamConfig, target: TargetConfig) -> float:
    """Integrated luminosity in ub^-1  (events = sigma[ub] * luminosity).

    Equivalent to the Fortran ``luminosity = charge / targetfac``
    (``targetfac = mass_amu/3.75914e6/(abundancy/100)*|cos(angle)|/thick_mg``)
    but derived from physical constants instead of a magic number.
    """
    n_electrons = beam.charge_mC * 1e-3 / E_CHARGE_C
    thick_g_cm2 = target.thickness_mg_cm2 / 1000.0
    thick_g_cm2 /= abs(math.cos(math.radians(target.angle_deg)))
    n_targets_per_cm2 = (thick_g_cm2 * N_AVOGADRO / target.mass_amu) * target.abundancy_pct / 100.0
    return n_electrons * n_targets_per_cm2 * 1e-30      # cm^2 -> ub


def sample_vertex(rng: np.random.Generator, n: int, beam: BeamConfig, target: TargetConfig):
    """Vertex (x, y, z) in cm and the raster offsets.  Mirrors ``generate`` in event.f."""
    nsig = 3.0
    x = _trunc_normal(rng, n, nsig) * beam.xwid + target.x_offset
    y = _trunc_normal(rng, n, nsig) * beam.ywid + target.y_offset
    p = beam.raster_pattern
    if p == 1:      # legacy bedpost square raster (cosine-distributed)
        rx = np.cos(rng.random(n) * np.pi) * beam.raster_1
        ry = np.cos(rng.random(n) * np.pi) * beam.raster_2
    elif p == 2:    # circular
        t = rng.random(n) * 2 * np.pi
        r = np.sqrt(rng.random(n)) * (beam.raster_2 - beam.raster_1) + beam.raster_1
        rx, ry = np.cos(t) * r, np.sin(t) * r
    elif p == 3:    # flat square
        rx = beam.raster_1 * (2 * rng.random(n) - 1)
        ry = beam.raster_2 * (2 * rng.random(n) - 1)
    else:
        rx = ry = np.zeros(n)
    z = (rng.random(n) - 0.5) * target.length_cm + target.z_offset
    return x + rx, y + ry, z


def sample_beam_energy(rng: np.random.Generator, n: int, beam: BeamConfig):
    width = beam.energy * beam.energy_spread_pct / 100.0
    return beam.energy + (rng.random(n) - 0.5) * width


def _trunc_normal(rng: np.random.Generator, n: int, nsig: float) -> np.ndarray:
    out = rng.standard_normal(n)
    bad = np.abs(out) > nsig
    while bad.any():
        out[bad] = rng.standard_normal(int(bad.sum()))
        bad = np.abs(out) > nsig
    return out


class EnergyLoss(Protocol):
    """Hook for target energy loss: returns MeV lost for each event."""
    def beam(self, batch, rng) -> np.ndarray: ...
    def scattered(self, batch, particle: str, rng) -> np.ndarray: ...
