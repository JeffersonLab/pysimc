import dataclasses

import numpy as np
import pytest

from pysimc.runner import run
from conftest import replace_run


def _key(r):
    return (r.n_tried, r.n_accepted, r.sum_w, r.sum_w2, r.hists["Q2"].counts.tolist())


def test_same_seed_is_reproducible(base_cfg):
    assert _key(run(base_cfg, workers=1)) == _key(run(base_cfg, workers=1))


def test_different_seed_differs(base_cfg):
    a, b = run(base_cfg, workers=1, seed=1), run(base_cfg, workers=1, seed=2)
    assert a.sum_w != b.sum_w


@pytest.mark.parametrize("workers", [2, 3])
def test_result_independent_of_worker_count(base_cfg, workers):
    """Bit-for-bit identical numbers for 1 vs N processes."""
    assert _key(run(base_cfg, workers=1)) == _key(run(base_cfg, workers=workers))


def test_accepted_mode_independent_of_worker_count(base_cfg):
    cfg = replace_run(base_cfg, mode="accepted", n_accepted=3000, chunk_trials=10_000)
    assert _key(run(cfg, workers=1)) == _key(run(cfg, workers=2))


def test_ntuple_columns_merge(base_cfg):
    cfg = dataclasses.replace(base_cfg, output=dataclasses.replace(
        base_cfg.output, ntuple=True, ntuple_columns=["Q2", "weight"]))
    r1, r2 = run(cfg, workers=1), run(cfg, workers=2)
    cols = r1.event_columns()
    assert len(cols["Q2"]) == r1.n_accepted
    np.testing.assert_array_equal(cols["Q2"], r2.event_columns()["Q2"])


def test_unported_physics_fails_readably_with_workers(base_cfg):
    cfg = dataclasses.replace(base_cfg, radiation=dataclasses.replace(base_cfg.radiation, model="mo_tsai"))
    with pytest.raises(NotImplementedError, match="radc.f"):
        run(cfg, workers=2)
