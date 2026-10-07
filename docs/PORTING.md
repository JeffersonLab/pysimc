# Porting guide: Fortran → pysimc

Port in the order below; each step is independently testable and nothing in `engine.py`
or `runner.py` needs to change.

## 0. Rules of thumb

* **Don't translate line by line.** Read what a routine *computes* and write it as a
  vectorised function of column arrays (`EventBatch`). Replace `if success` early-exits
  with boolean masks.
* **Keep a Fortran reference run** for each ported piece (central-event numbers from
  `report`, a few histograms) and add a regression test with a tolerance.
* Units follow the Fortran: MeV, MeV/c, rad internally, δ in %, slopes dimensionless.
  Config files use degrees and mrad (as the legacy inputs do).
* Coordinates: lab z = beam, x = down, y = beam-left; `xptar` out-of-plane, `yptar` in-plane.

## 1. Spectrometers  (`spectrometers/`)

| Fortran | What to port |
|---|---|
| `hms/mc_hms.f`, `mc_hms_hut.f`, `mc_hms_coll.f`, `apertures_hms.inc` | forward transport, collimator/aperture checks, hut multiple scattering + resolution |
| `hms/mc_hms_recon.f`, `recon_cosy.dat` | reconstruction (for resolutions and `recon` columns) |
| `shms/`, `sos/`, `hrsl/`, `hrsr/`, `calo/` | same pattern |

* Parse `forward_cosy.dat` / `recon_cosy.dat` into `PolynomialMatrix` objects (already implemented
  and tested).
* Write `MatrixSpectrometer(Spectrometer)` and register it under the existing names
  (`@SPECTROMETERS.register("hms")`, replacing the box stub).
* Extend the `accept` contract if needed to return reconstructed target quantities; add
  `e_delta_recon` etc. columns so resolution studies work.
* Spectrometer offsets (`spec.*.offset.*`) belong here; the config fields already exist.

## 2. Target energy loss and Coulomb correction  (`target.py`)

`target.f`, `enerloss_new.f`, `trip_thru_target` in `event.f`. Implement the `EnergyLoss`
protocol and apply it in the reaction (beam) and in the engine (outgoing particles). Add
`using_eloss`, `using_coulomb`, `correct_eloss` to the config as typed booleans.

## 3. Radiative corrections  (`radiative.py`)

`radc.f`, `brem.f`, `init.f` (`radc_init`, `radc_init_ev`), `radc.inc`.
Implement `MoTsai(RadiativeCorrections)`: sample photon energies, shift `Ein`/`Eprime`,
update `Q2`, `nu`, multiply `weight`. It must also handle the `Egamma` generation limits.
Validate against `report` output (`hardcorfac`, `etatzai`, `g_int`, `g_ext`).

## 4. Reactions  (`reactions/`)

Replace entries from `stubs.py` one at a time (suggested order is in that file).
For a reaction with more random variables than the two-body elastic case, `generate` draws
them with a known density and puts `density^-1 × Jacobian × dσ` into `weight`.
The Fortran `generate`, `complete_ev`, `complete_main` and `jacobians.f` define these.
Spectral functions (`sf_lookup.f`, `benharsf*`) become lookup tables loaded once per process.

## 5. Form factors and cross-section options

`physics.py` has a dipole placeholder. Port `physics_proton.f` (Galster, Bosted, deForest
CC1/CC2) behind `form_factors(q2) -> (GE, GM)` and expose the choice in `ReactionConfig.options`.

## 6. Legacy options

`pysimc convert` prints every `.inp` key it ignored. Work through that list as features land:
add a typed field in `config.py` and a mapping line in `legacy.py`.

## 7. Output

`.ntuple` columns are an arbitrary subset of `EventBatch` columns. Add ROOT (`uproot`) or
HDF5 writers in `output.py` behind optional extras if downstream analysis needs them.

## Validation checklist per ported piece

1. Unit test of the formula against an independent calculation or a number from the Fortran.
2. Central-event numbers match `report` (Q2, ν, W, δ, xptar, yptar).
3. Integrated cross section within statistical error of a Fortran run with the same input.
4. `pytest tests/test_parallel.py` still passes (determinism is a feature).
