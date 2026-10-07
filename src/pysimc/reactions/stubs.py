"""Reactions still to be ported, one per Fortran physics module.

Each stub documents the source file to port from.  Replace a stub by a real
subclass of ``Reaction`` registered under the same name; nothing else changes.

Suggested order (increasing difficulty):
  1. phase_space      - no physics, validates acceptance models
  2. eep_quasi_elastic - needs spectral functions (sf_lookup.f, benharsf*)
  3. pion / kaon      - physics_pion.f, physics_kaon.f (+ decay in-flight)
  4. delta, rho       - physics_delta.f, rho_physics.f, generate_rho.f
  5. semi_inclusive   - semi_physics.f, F1F2*.f structure functions, fragmentation
"""

from .base import not_ported

not_ported("phase_space", "pure phase-space generation (doing_phsp)", "event.f:generate")
not_ported("eep_quasi_elastic", "A(e,e'p), D(e,e'p) with spectral function", "physics_proton.f, sf_lookup.f, nuclear_elastic.f")
not_ported("pion", "A(e,e'pi) electroproduction", "physics_pion.f, event.f:complete_ev")
not_ported("kaon", "A(e,e'K) electroproduction", "physics_kaon.f")
not_ported("delta", "H(e,e'p)pi / Delta production", "physics_delta.f")
not_ported("rho", "H(e,e'rho)", "rho_physics.f, generate_rho.f, rho_decay.f")
not_ported("semi_inclusive", "SIDIS A(e,e'h)X", "semi_physics.f, F1F2*.f")
