"""Command line interface:  pysimc run | convert | list"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from .config import SimConfig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pysimc", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a simulation from a .toml or legacy .inp file")
    r.add_argument("config")
    r.add_argument("-j", "--workers", type=int, default=None, help="processes (default: all CPUs)")
    r.add_argument("--seed", type=int, default=None, help="override run.seed")
    r.add_argument("--trials", type=int, default=None, help="override: fixed number of trials")
    r.add_argument("-o", "--output", default=None, help="output directory (default: config's)")
    r.add_argument("--no-files", action="store_true", help="print the report only")

    c = sub.add_parser("convert", help="convert a legacy .inp file to TOML")
    c.add_argument("inp")
    c.add_argument("-o", "--output", default=None)

    sub.add_parser("list", help="list registered reactions, spectrometers, radiation models")

    args = ap.parse_args(argv)
    try:
        return {"run": _run, "convert": _convert, "list": _list}[args.cmd](args)
    except (NotImplementedError, ValueError, KeyError, FileNotFoundError) as exc:
        print(f"pysimc: error: {exc}", file=sys.stderr)
        return 2


def _run(args) -> int:
    from .output import format_report, write_outputs
    from .runner import run

    cfg = SimConfig.from_file(args.config)
    if args.trials is not None:
        cfg = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, mode="trials", n_trials=args.trials))
    res = run(cfg, workers=args.workers, seed=args.seed)
    print(format_report(cfg, res))
    if not args.no_files:
        print("wrote", write_outputs(cfg, res, args.output))
    return 0


def _convert(args) -> int:
    from .legacy import load_legacy_inp, to_toml
    text = to_toml(load_legacy_inp(args.inp))
    SimConfig.from_dict(__import__("tomllib").loads(text))     # round-trip check
    if args.output:
        Path(args.output).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


def _list(args) -> int:
    from .radiative import RADIATION_MODELS
    from .reactions import REACTIONS
    from .spectrometers import SPECTROMETERS
    from .reactions.base import NotPortedReaction
    from .radiative import MoTsai
    from .spectrometers.base import BoxSpectrometer

    def tag(reg, stub):
        return ", ".join(n + (" (stub)" if stub(reg.get(n)) else "") for n in reg.names())

    print("reactions     :", tag(REACTIONS, lambda c: issubclass(c, NotPortedReaction)))
    print("spectrometers :", tag(SPECTROMETERS, lambda c: c is not BoxSpectrometer and issubclass(c, BoxSpectrometer)),
          " [stub = box acceptance only]")
    print("radiation     :", tag(RADIATION_MODELS, lambda c: c is MoTsai))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
