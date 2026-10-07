"""pysimc -- vectorised, parallel Monte Carlo for coincidence electron scattering."""

from .config import SimConfig
from .engine import Simulation
from .results import Result
from .runner import run

__version__ = "0.1.0"
__all__ = ["SimConfig", "Simulation", "Result", "run"]
