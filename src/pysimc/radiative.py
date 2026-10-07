"""Radiative-correction models.

Interface: ``apply(batch, rng)`` returns the batch with photon emission folded
in (modified Ein/Eprime/Q2 ... and weights).  The Fortran implementation is
spread over ``radc.f``, ``brem.f`` and ``init.f`` (Mo-Tsai / Ent et al.
prescriptions selected by rad_flag/extrad_flag/intcor_mode); port it as one or
more ``RadiativeCorrections`` subclasses.  Reactions never need to know.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import SimConfig
from .events import EventBatch
from .registry import Registry

RADIATION_MODELS: Registry[type["RadiativeCorrections"]] = Registry("radiation model")


class RadiativeCorrections(ABC):
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    @abstractmethod
    def apply(self, batch: EventBatch, rng: np.random.Generator) -> EventBatch:
        ...


@RADIATION_MODELS.register("none")
class NoRadiation(RadiativeCorrections):
    def apply(self, batch, rng):
        return batch


@RADIATION_MODELS.register("mo_tsai")
class MoTsai(RadiativeCorrections):
    """TODO: port radc.f / brem.f (soft-photon exponentiation, hard photon sampling,
    internal + external Bremsstrahlung, Schwinger term)."""

    def __init__(self, cfg):
        raise NotImplementedError(
            "radiation model 'mo_tsai' is not ported yet (Fortran: radc.f, brem.f); "
            "set radiation.model = \"none\" to run without radiative corrections")

    def apply(self, batch, rng):        # pragma: no cover
        raise NotImplementedError


def build_radiation(cfg: SimConfig) -> RadiativeCorrections:
    return RADIATION_MODELS.get(cfg.radiation.model)(cfg)
