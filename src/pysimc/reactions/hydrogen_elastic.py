"""H(e,e'p) elastic: the reference reaction for the skeleton.

Two-body kinematics fix the whole event from the electron direction, so only
the electron slopes (xptar, yptar), beam energy and vertex are random.  The
hadron-arm coordinates follow from momentum conservation and are tested
against the hadron spectrometer acceptance like any other reaction.

Weight:  sigma(ub/sr) * dx*dy*(1+x^2+y^2)^(-3/2)  with (dx, dy) the width of the
uniform generation box in electron slopes -- the sampling density is
1/(dx*dy), the physical measure is dOmega, hence the Jacobian.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from ..constants import MP
from ..events import EventBatch
from ..kinematics import (BEAM, arm_central_theta, direction_to_slopes,
                          slopes_to_direction, solid_angle_jacobian)
from ..physics import dipole_form_factors, rosenbluth_xsec
from ..target import sample_beam_energy, sample_vertex
from .base import REACTIONS, Reaction


@REACTIONS.register("hydrogen_elastic")
class HydrogenElastic(Reaction):
    def __init__(self, cfg: SimConfig):
        super().__init__(cfg)
        e, p = cfg.electron_arm, cfg.hadron_arm
        self.th_e, self.th_p = arm_central_theta(e), arm_central_theta(p)
        self.x_lo, self.x_hi = (v * 1e-3 for v in e.xptar_mrad)
        self.y_lo, self.y_hi = (v * 1e-3 for v in e.yptar_mrad)
        self.gen_area = (self.x_hi - self.x_lo) * (self.y_hi - self.y_lo)

    def generate(self, rng: np.random.Generator, n: int) -> EventBatch:
        cfg = self.cfg
        e_in = sample_beam_energy(rng, n, cfg.beam)
        vx, vy, vz = sample_vertex(rng, n, cfg.beam, cfg.target)
        xp = self.x_lo + rng.random(n) * (self.x_hi - self.x_lo)
        yp = self.y_lo + rng.random(n) * (self.y_hi - self.y_lo)

        # electron
        u_e = slopes_to_direction(self.th_e, cfg.electron_arm.side, xp, yp)
        cos_t = u_e @ BEAM
        theta_e = np.arccos(np.clip(cos_t, -1, 1))
        e_prime = e_in / (1.0 + (e_in / MP) * (1.0 - cos_t))

        # momentum transfer and recoil proton (target at rest)
        q_vec = e_in[:, None] * BEAM - e_prime[:, None] * u_e
        p_p = np.linalg.norm(q_vec, axis=1)
        nu = e_in - e_prime
        q2 = 2.0 * e_in * e_prime * (1.0 - cos_t)          # massless electron
        u_p = q_vec / p_p[:, None]
        pxp, pyp = direction_to_slopes(self.th_p, cfg.hadron_arm.side, u_p)

        sigma = rosenbluth_xsec(e_in, theta_e, q2, dipole_form_factors)
        weight = sigma * self.gen_area * solid_angle_jacobian(xp, yp)

        return EventBatch(n, {
            "weight": weight,
            "Ein": e_in, "Eprime": e_prime, "theta_e": theta_e,
            "nu": nu, "Q2": q2,
            "W": np.sqrt(np.maximum(MP ** 2 + 2.0 * MP * nu - q2, 0.0)),
            "Em": np.zeros(n), "Pm": np.zeros(n),          # exact for two-body elastic
            "p_mom": p_p,
            "e_delta": (e_prime / cfg.electron_arm.momentum - 1.0) * 100.0,
            "e_xptar": xp, "e_yptar": yp,
            "p_delta": (p_p / cfg.hadron_arm.momentum - 1.0) * 100.0,
            "p_xptar": pxp, "p_yptar": pyp,
            "vx": vx, "vy": vy, "vz": vz,
        })
