# pysimc

A vectorised, multi-process Monte Carlo for coincidence electron scattering,
designed as the successor to the Fortran SIMC (`simc_gfortran`). It reproduces
SIMC's *workflow* (generate → radiate → spectrometer acceptance → weight →
histogram), not its code layout.

**Status: skeleton.** One reaction works end to end (H(e,e'p) elastic). Every other
reaction, the radiative corrections and the spectrometer optics are registered
stubs that fail at start-up with a message naming the Fortran file to port.
See [`docs/PORTING.md`](docs/PORTING.md).

## Quick start

```bash
pip install -e ".[dev]"
pytest                                        # 32 tests
pysimc list                                   # what exists / what is a stub
pysimc run examples/hydrogen_elastic.toml -j 4
pysimc convert examples/legacy/eep_hydrogen_q8.inp -o q8.toml   # legacy .inp -> TOML
```

Output goes to `output.directory`: `report.txt`, `summary.json`, `config_used.json`,
`histograms.npz`, and `events.npz` when `output.ntuple = true`.

From Python:

```python
from pysimc import SimConfig, run
cfg = SimConfig.from_file("examples/hydrogen_elastic.toml")
res = run(cfg, workers=4, seed=1)
print(res.sigma_ub, res.sigma_err_ub, res.n_accepted)
```

## How it differs from the Fortran

| Fortran SIMC | pysimc |
|---|---|
| One event at a time through global common blocks | Batches of events as numpy column arrays (`EventBatch`), no globals |
| Serial; one RANLUX stream | Process pool; one independent `SeedSequence` stream **per chunk** |
| Results depend on run history | Results are **bit-identical for any worker count** for a given seed and `chunk_trials` |
| CTP `.inp`, silent on typos | Typed TOML config, unknown keys are errors; legacy `.inp` import via `pysimc convert` |
| Reaction chosen by `doing_*` flags and big `if` chains | Plugin registries: `REACTIONS`, `SPECTROMETERS`, `RADIATION_MODELS` |
| HBOOK / ntuple | Mergeable numpy histograms + npz event columns (ROOT/HDF5 writers can be added) |
| Magic luminosity constant `3.75914e6` | Computed from CODATA constants (agrees to 0.2%, tested) |

## Layout

```
src/pysimc/
  config.py        typed dataclass config + strict loader (TOML, legacy .inp)
  legacy.py        CTP .inp parser, option mapping, TOML writer
  runner.py        chunking, process pool, ordered merge  <- parallelism lives here
  engine.py        Simulation: generate -> radiate -> accept -> cuts -> accumulate
  events.py        EventBatch (structure-of-arrays) and column vocabulary
  reactions/       Reaction interface; hydrogen_elastic (working); stubs.py (to port)
  spectrometers/   Spectrometer interface; box acceptance; PolynomialMatrix (COSY building block)
  radiative.py     RadiativeCorrections interface; none / mo_tsai (stub)
  target.py        beam & vertex sampling, luminosity; energy-loss interface (stub)
  physics.py       Mott, Rosenbluth, (placeholder) dipole form factors
  kinematics.py    slopes <-> lab directions, solid-angle Jacobian
  results.py       Result / Hist1D: additive accumulators
  output.py        report, summary.json, npz files
tests/             kinematics, physics, 2-body closure, MC-vs-quadrature, parallel determinism, config
```

## Weights and normalisation

Every reaction returns `weight` in µb such that `sum(weight of accepted) / n_generated`
is the integrated cross section inside the acceptance. Multiply by
`luminosity_per_ub` (target.py) for expected counts. Reactions are responsible for the
generation-volume and Jacobian factors, so the engine stays reaction-agnostic.

## Parallelism notes

* Work unit = chunk of `run.chunk_trials` trials. Keep it large (≥ 1e5) so process overhead is negligible.
* Changing `chunk_trials` changes the random streams, so results change statistically
  (not the worker count).
* `run.mode = "accepted"` is the legacy `ngen > 0` behaviour; it runs in waves of 16 chunks.
* On this 2-core test machine: 2.1 M → 3.8 M trials/s going from 1 to 2 workers
  (elastic reaction, no ntuple). A serial Fortran-style event loop in pure Python
  would be ~50× slower; vectorisation is where most of the gain comes from.
