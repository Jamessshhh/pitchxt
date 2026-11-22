import numpy as np
import pandas as pd

from pitchxt import fit_xt, xt_lookup, add_xt


def _synthetic(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 120, n)
    y0 = rng.uniform(0, 80, n)
    x1 = np.clip(x0 + rng.normal(8, 10, n), 0, 120)
    y1 = np.clip(y0 + rng.normal(0, 8, n), 0, 80)
    shot = rng.random(n) < np.clip((x0 - 70) / 50, 0, 0.5)
    goal = shot & (rng.random(n) < 0.15)
    success = (rng.random(n) < 0.8) & ~shot
    return pd.DataFrame({
        "kind": np.where(shot, "shot", "move"),
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "success": success, "goal": goal,
    })


def test_grid_shape_and_range():
    g = fit_xt(_synthetic())
    assert g.shape == (12, 16)
    assert (g >= 0).all() and g.max() < 1


def test_threat_rises_toward_goal():
    g = fit_xt(_synthetic())
    assert g[:, -3:].mean() > g[:, :3].mean()
    assert xt_lookup(g, 115, 40) > xt_lookup(g, 5, 40)


def test_failed_moves_never_gain():
    a = _synthetic()
    out = add_xt(a, fit_xt(a))
    failed = out[(out.kind == "move") & (~out.success)]
    assert (failed.xt_gain <= 0).all()
    assert out[out.kind == "shot"].xt_gain.isna().all()