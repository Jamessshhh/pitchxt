import numpy as np

from pitchxt.pitch_control import make_grid
from pitchxt.value import xt_on_control_grid


def _grid():
    """Synthetic xT that rises toward x = 120 (the attacking end)."""
    xs = (np.arange(16) + 0.5) / 16 * 120
    return np.tile(xs / 120, (12, 1)) ** 2


def test_home_values_rise_toward_plus_x():
    xs, ys = make_grid(40, 20)
    v = xt_on_control_grid(_grid(), xs, ys, "home")
    assert v[:, -1].mean() > v[:, 0].mean()


def test_away_view_is_rotated():
    xs, ys = make_grid(40, 20)
    g = _grid()
    home = xt_on_control_grid(g, xs, ys, "home")
    away = xt_on_control_grid(g, xs, ys, "away")
    assert away[:, 0].mean() > away[:, -1].mean()          # away attacks toward -x
    assert np.allclose(home, away[::-1, ::-1])             # exact 180 degree rotation


def test_shape_matches_control_grid():
    xs, ys = make_grid(50, 32)
    assert xt_on_control_grid(_grid(), xs, ys).shape == (32, 50)