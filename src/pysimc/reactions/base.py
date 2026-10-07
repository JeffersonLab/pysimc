"""Reaction interface.

A ``Reaction`` owns the physics of one process: how events are generated
(which variables are drawn at random, with which phase-space density), the
kinematics, and the cross section.  The engine only needs the returned
``EventBatch``; see ``events.py`` for the column vocabulary.

Contract
--------
``generate(rng, n)`` returns n events with
  * physical kinematics for BOTH arms: ``e_delta, e_xptar, e_yptar, p_delta,
    p_xptar, p_yptar`` so the engine can apply spectrometer acceptance;
  * ``weight`` = (differential cross section) x (generation phase-space volume
    / Jacobian) in ub, so that  sum(weight over accepted) / n_generated  is the
    integrated cross section in ub inside the acceptance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import SimConfig
from ..events import EventBatch
from ..registry import Registry

REACTIONS: Registry[type["Reaction"]] = Registry("reaction")


class Reaction(ABC):
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    @abstractmethod
    def generate(self, rng: np.random.Generator, n: int) -> EventBatch:
        ...


class NotPortedReaction(Reaction):
    """Declared so the name is discoverable; raises until implemented."""

    fortran_source: str = ""
    description: str = ""

    def __init__(self, cfg: SimConfig):
        # fail at start-up, not half-way through a long run
        raise NotImplementedError(
            f"reaction {cfg.reaction.name!r} is not ported yet "
            f"({self.description}). Port from Fortran: {self.fortran_source}"
        )

    def generate(self, rng, n):          # pragma: no cover
        raise NotImplementedError


def not_ported(name: str, description: str, fortran_source: str) -> type[NotPortedReaction]:
    cls = type(f"{name.title().replace('_', '')}Reaction", (NotPortedReaction,),
               {"description": description, "fortran_source": fortran_source,
                "__doc__": f"{description} (stub; see {fortran_source})"})
    REACTIONS.register(name)(cls)
    return cls
