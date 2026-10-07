"""Spectrometer interface and the default box-acceptance model."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import ArmConfig
from ..registry import Registry

SPECTROMETERS: Registry[type["Spectrometer"]] = Registry("spectrometer")


class Spectrometer(ABC):
    """Maps target-plane coordinates of a batch of tracks to accept/reject.

    A full model (see ``MatrixSpectrometer`` below) will also return focal-plane
    coordinates, apply multiple scattering / detector smearing and reconstruct
    target quantities.  All methods are vectorised over the batch.
    """

    def __init__(self, arm: ArmConfig):
        self.arm = arm

    @abstractmethod
    def accept(self, delta: np.ndarray, xptar: np.ndarray, yptar: np.ndarray,
               vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
        """Boolean mask of tracks accepted.  delta in %, slopes dimensionless."""


@SPECTROMETERS.register("box")
class BoxSpectrometer(Spectrometer):
    """Rectangular acceptance in (delta, xptar, yptar) from the arm config.

    This is what the legacy ``SPedge`` limits describe.  It is the fallback
    for every named spectrometer until its transport model is ported.
    """

    def accept(self, delta, xptar, yptar, vx, vy, vz, rng):
        a = self.arm
        return (
            (delta >= a.delta[0]) & (delta <= a.delta[1])
            & (xptar >= a.xptar_mrad[0] * 1e-3) & (xptar <= a.xptar_mrad[1] * 1e-3)
            & (yptar >= a.yptar_mrad[0] * 1e-3) & (yptar <= a.yptar_mrad[1] * 1e-3)
        )


def _named_box(name: str, fortran_src: str):
    """Placeholder for a spectrometer whose COSY transport is not ported yet."""
    cls = type(
        f"{name.upper()}Spectrometer",
        (BoxSpectrometer,),
        {"__doc__": f"{name.upper()}: box acceptance for now.  TODO: port {fortran_src} "
                    f"(COSY forward/reconstruction matrices, apertures, hut resolution)."},
    )
    SPECTROMETERS.register(name)(cls)
    return cls


HMS = _named_box("hms", "hms/mc_hms.f")
SOS = _named_box("sos", "sos/mc_sos.f")
SHMS = _named_box("shms", "shms/mc_shms.f")
HRSL = _named_box("hrsl", "hrsl/mc_hrsl.f")
HRSR = _named_box("hrsr", "hrsr/mc_hrsr.f")
CALO = _named_box("calo", "calo/mc_calo.f")


def build_spectrometer(arm: ArmConfig) -> Spectrometer:
    return SPECTROMETERS.get(arm.spectrometer)(arm)
