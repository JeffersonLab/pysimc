"""Reader for legacy SIMC ``.inp`` (CTP) files  ->  native config dict.

    pysimc convert infiles/eep_hydrogen_q8.inp -o eep_hydrogen_q8.toml

Only the options the skeleton understands are mapped.  Everything else is
reported with a warning rather than dropped silently; extend ``_MAP`` /
``load_legacy_inp`` as physics is ported.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

# Fortran electron_arm / hadron_arm codes (dbase.f): spectrometer, side of beam
ARM_CODES = {
    1: ("hms", "right"), 2: ("sos", "left"), 3: ("hrsr", "right"), 4: ("hrsl", "left"),
    5: ("shms", "left"), 7: ("calo", "right"), 8: ("calo", "left"),
}

_KV = re.compile(r"^\s*([A-Za-z0-9_%.()]+)\s*=\s*(.+?)\s*$")


def parse_ctp(path: Path) -> dict[str, Any]:
    """Flat dict with lower-case keys, '%' -> '.'.  Section markers are ignored."""
    out: dict[str, Any] = {}
    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or line.lower().startswith(("begin", "end")):
            continue
        m = _KV.match(line)
        if m:
            out[m.group(1).lower().replace("%", ".")] = _value(m.group(2))
    return out


def _value(tok: str) -> Any:
    tok = tok.strip().strip(",")
    if tok[:1] in "'\"":
        return tok.strip("'\"")
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        return tok


def load_legacy_inp(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    kv = parse_ctp(path)
    extra = kv.get("extra_dbase_file")
    if extra:
        for cand in (path.parent / extra, path.parent / f"{extra}.inp"):
            if cand.is_file():
                kv = {**parse_ctp(cand), **kv}
                break
        else:
            warnings.warn(f"extra_dbase_file {extra!r} not found next to {path.name}")
    used: set[str] = {"extra_dbase_file"}

    def g(key, default=None):
        used.add(key)
        return kv.get(key, default)

    def arm(code_key: str, who: str) -> dict[str, Any]:
        code = int(g(code_key, 1))
        if code not in ARM_CODES:
            raise ValueError(f"{code_key}={code} is not a supported spectrometer")
        name, side = ARM_CODES[code]
        s = "e" if who == "e" else "p"
        return {
            "spectrometer": name, "side": side,
            "momentum": g(f"spec.{s}.p", 0.0),
            "theta_deg": abs(g(f"spec.{s}.theta", 0.0)),
            "delta": [g(f"spedge.{s}.delta.min", -10.0), g(f"spedge.{s}.delta.max", 10.0)],
            "xptar_mrad": [g(f"spedge.{s}.xptar.min", -100.0), g(f"spedge.{s}.xptar.max", 100.0)],
            "yptar_mrad": [g(f"spedge.{s}.yptar.min", -50.0), g(f"spedge.{s}.yptar.max", 50.0)],
            "offset_x_cm": g(f"spec.{s}.offset.x", 0.0), "offset_y_cm": g(f"spec.{s}.offset.y", 0.0),
            "offset_xptar_mrad": g(f"spec.{s}.offset.xptar", 0.0),
            "offset_yptar_mrad": g(f"spec.{s}.offset.yptar", 0.0),
        }

    ngen = int(g("ngen", 5000))
    run = ({"mode": "accepted", "n_accepted": ngen} if ngen > 0
           else {"mode": "trials", "n_trials": -ngen})

    A = g("targ.a", 1.0)
    cfg: dict[str, Any] = {
        "run": run,
        "beam": {
            "energy": g("ebeam", 0.0), "energy_spread_pct": g("debeam", 0.0),
            "charge_mC": g("exper.charge", 1.0),
            "xwid": g("gen.xwid", 0.0), "ywid": g("gen.ywid", 0.0),
            "raster_pattern": int(g("targ.fr_pattern", 0)),
            "raster_1": g("targ.fr1", 0.0), "raster_2": g("targ.fr2", 0.0),
        },
        "target": {
            "A": A, "Z": g("targ.z", 1.0), "mass_amu": g("targ.mass_amu", 1.0),
            "rho": g("targ.rho", 0.0723), "thickness_mg_cm2": g("targ.thick", 100.0),
            "angle_deg": g("targ.angle", 0.0), "abundancy_pct": g("targ.abundancy", 100.0),
            "x_offset": g("targ.xoffset", 0.0), "y_offset": g("targ.yoffset", 0.0),
            "z_offset": g("targ.zoffset", 0.0),
        },
        "electron_arm": arm("electron_arm", "e"),
        "hadron_arm": arm("hadron_arm", "p"),
        "reaction": {"name": _reaction_name(kv, A, g)},
        "radiation": {"model": "mo_tsai" if int(g("using_rad", 0)) else "none"},
    }

    spect_mode = int(g("spect_mode", 0))
    if spect_mode == -1:
        cfg["require_arms"] = ["p"]
    elif spect_mode == -2:
        cfg["require_arms"] = ["e"]
    elif spect_mode == 1:
        raise ValueError("spect_mode=1 (no spectrometers) is not supported")

    em_lo, em_hi = g("cuts.em.min", 0.0), g("cuts.em.max", 0.0)
    if (em_lo, em_hi) != (0.0, 0.0):          # legacy: 0,0 means "wide open"
        cfg["cuts"] = {"Em": [em_lo, em_hi]}

    ignored = sorted(k for k in kv if k not in used and not k.startswith("debug"))
    if ignored:
        warnings.warn(f"{path.name}: {len(ignored)} legacy option(s) not mapped yet: {ignored}",
                      stacklevel=2)
    return cfg


def _reaction_name(kv, A, g) -> str:
    flag = lambda k: int(g(k, 0)) == 1
    if flag("doing_phsp"):
        return "phase_space"
    if flag("doing_semi") or flag("doing_semipi") or flag("doing_semika"):
        return "semi_inclusive"
    if flag("doing_pion"):
        return "pion"
    if flag("doing_kaon"):
        return "kaon"
    if flag("doing_delta"):
        return "delta"
    if flag("doing_rho"):
        return "rho"
    return "hydrogen_elastic" if round(A) == 1 else "eep_quasi_elastic"


# -- minimal TOML writer (stdlib has a reader only) -------------------------------

def to_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    _emit(data, [], lines)
    return "\n".join(lines).strip() + "\n"


def _scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_scalar(x) for x in v) + "]"
    return json.dumps(str(v))


def _emit(d: dict[str, Any], prefix: list[str], lines: list[str]) -> None:
    scalars = {k: v for k, v in d.items()
               if v is not None and not isinstance(v, dict)
               and not (isinstance(v, list) and v and isinstance(v[0], dict))}
    if prefix and (scalars or not any(isinstance(v, dict) for v in d.values())):
        lines += ["", f"[{'.'.join(prefix)}]"]
    for k, v in scalars.items():
        lines.append(f"{k} = {_scalar(v)}")
    for k, v in d.items():
        if isinstance(v, dict):
            _emit(v, prefix + [k], lines)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            for item in v:
                lines += ["", f"[[{'.'.join(prefix + [k])}]]"]
                lines += [f"{kk} = {_scalar(vv)}" for kk, vv in item.items()]
