"""Typed configuration.

One ``SimConfig`` describes a whole run.  It is built from a TOML file (the
native format) or from a legacy SIMC ``.inp`` file (see ``legacy.py``).
Unknown keys are an error, so typos in a config can no longer be silently
ignored the way they were in the CTP parser.

Units in config files (kept the same as the legacy inputs, to ease migration):
    energies/momenta MeV, (MeV/c)   angles deg   delta %   xptar/yptar mrad
    target thickness mg/cm^2        density g/cm^3   charge mC
"""

from __future__ import annotations

import dataclasses
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Range = tuple[float, float]


@dataclass
class RunConfig:
    mode: str = "trials"          # "trials": fixed number of generated events
                                  # "accepted": run until n_accepted events pass
    n_trials: int = 1_000_000
    n_accepted: int = 5000
    max_trials: int = 500_000_000  # safety stop for mode="accepted"
    batch_size: int = 50_000       # vectorised batch (memory/cache knob)
    chunk_trials: int = 200_000    # unit of parallel work AND of RNG streams
    seed: int | None = None        # None -> fresh entropy (recorded in output)


@dataclass
class BeamConfig:
    energy: float = 0.0            # MeV
    energy_spread_pct: float = 0.0  # full width, % of energy (legacy dEbeam)
    charge_mC: float = 1.0
    xwid: float = 0.0              # intrinsic beam sigma, cm
    ywid: float = 0.0
    raster_pattern: int = 0        # 0 none, 1 legacy bedpost, 2 circular, 3 flat
    raster_1: float = 0.0          # cm (half-widths, or inner radius)
    raster_2: float = 0.0          # cm (half-widths, or outer radius)


@dataclass
class TargetConfig:
    A: float = 1.0
    Z: float = 1.0
    mass_amu: float = 1.007276
    rho: float = 0.0723            # g/cm^3
    thickness_mg_cm2: float = 100.0
    angle_deg: float = 0.0         # target tilt (solid targets)
    abundancy_pct: float = 100.0
    x_offset: float = 0.0          # cm
    y_offset: float = 0.0
    z_offset: float = 0.0

    @property
    def length_cm(self) -> float:
        return self.thickness_mg_cm2 / 1000.0 / self.rho


@dataclass
class ArmConfig:
    spectrometer: str = "hms"
    momentum: float = 0.0          # central momentum, MeV/c
    theta_deg: float = 0.0         # central angle (always positive)
    side: str = "right"            # "left" or "right" of the beam, looking downstream
    delta: Range = (-10.0, 10.0)   # acceptance edges / generation limits, %
    xptar_mrad: Range = (-100.0, 100.0)  # out-of-plane slope
    yptar_mrad: Range = (-50.0, 50.0)    # in-plane slope
    offset_x_cm: float = 0.0       # reserved: spectrometer offsets (see spectrometers/)
    offset_y_cm: float = 0.0
    offset_xptar_mrad: float = 0.0
    offset_yptar_mrad: float = 0.0


@dataclass
class ReactionConfig:
    name: str = "hydrogen_elastic"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class RadiationConfig:
    model: str = "none"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class HistogramConfig:
    name: str = ""
    var: str = ""
    bins: int = 100
    range: Range = (0.0, 1.0)


@dataclass
class OutputConfig:
    directory: str = "results"
    ntuple: bool = False           # store accepted events as columns (npz)
    ntuple_columns: list[str] = field(default_factory=list)  # empty -> all
    histograms: list[HistogramConfig] = field(default_factory=list)


@dataclass
class SimConfig:
    run: RunConfig = field(default_factory=RunConfig)
    beam: BeamConfig = field(default_factory=BeamConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    electron_arm: ArmConfig = field(default_factory=ArmConfig)
    hadron_arm: ArmConfig = field(default_factory=ArmConfig)
    reaction: ReactionConfig = field(default_factory=ReactionConfig)
    radiation: RadiationConfig = field(default_factory=RadiationConfig)
    # Which arms must accept the event.  ["e","p"] = coincidence;
    # ["e"] = singles in the electron arm (legacy spect_mode = -2).
    require_arms: list[str] = field(default_factory=lambda: ["e", "p"])
    # Cuts on any event column: {"Em": [min, max]}.  Replaces the hard-coded
    # Em cut of the Fortran code.
    cuts: dict[str, Range] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimConfig":
        return _build(cls, data, path="")

    @classmethod
    def from_file(cls, path: str | Path) -> "SimConfig":
        path = Path(path)
        if path.suffix == ".toml":
            with open(path, "rb") as fh:
                return cls.from_dict(tomllib.load(fh))
        if path.suffix == ".inp":
            from .legacy import load_legacy_inp
            return cls.from_dict(load_legacy_inp(path))
        raise ValueError(f"unsupported config type: {path.suffix} (use .toml or .inp)")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        if self.run.mode not in ("trials", "accepted"):
            raise ValueError("run.mode must be 'trials' or 'accepted'")
        if self.beam.energy <= 0:
            raise ValueError("beam.energy must be > 0")
        for name, arm in (("electron_arm", self.electron_arm), ("hadron_arm", self.hadron_arm)):
            if arm.side not in ("left", "right"):
                raise ValueError(f"{name}.side must be 'left' or 'right'")
            if arm.momentum <= 0:
                raise ValueError(f"{name}.momentum must be > 0")
        if not set(self.require_arms) <= {"e", "p"} or not self.require_arms:
            raise ValueError("require_arms must be a non-empty subset of ['e','p']")
        if self.run.chunk_trials < 1 or self.run.batch_size < 1:
            raise ValueError("chunk_trials and batch_size must be >= 1")


# ---------------------------------------------------------------------------
# generic dict -> dataclass builder with strict key checking
# ---------------------------------------------------------------------------

def _build(cls: type, data: dict[str, Any], path: str):
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config key(s) in [{path or 'top level'}]: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        kwargs[name] = _coerce(hints[name], value, f"{path}.{name}".lstrip("."))
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, path: str):
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)
    if origin in (list, typing.List):
        (item,) = typing.get_args(tp)
        return [_coerce(item, v, path) for v in value]
    if origin is tuple:
        return tuple(float(v) for v in value)
    if origin is dict:
        args = typing.get_args(tp)
        if len(args) == 2 and typing.get_origin(args[1]) is tuple:
            return {k: tuple(float(x) for x in v) for k, v in value.items()}
        return dict(value)
    if tp is float:
        return float(value)
    if tp is int:
        return int(value)
    return value
