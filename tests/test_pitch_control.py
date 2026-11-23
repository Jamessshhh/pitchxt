import numpy as np

from pitchxt.pitch_control import make_grid, pitch_control_xy


def _still(*xy):
    return np.array([[x, y, 0.0, 0.0] for x, y in xy])


def _at(pc, xs, ys, x, y):
    return pc[np.abs(ys - y).argmin(), np.abs(xs - x).argmin()]


def test_probabilities_sum_to_about_one():
    xs, ys = make_grid(30, 20)
    pa, pd_ = pitch_control_xy(_still((-10, 0), (0, 10)), _still((10, 0), (20, -5)), (0, 0), xs, ys)
    assert (pa >= 0).all() and (pd_ >= 0).all()
    assert (pa + pd_ > 0.98).all() and (pa + pd_ < 1.0001).all()


def test_each_team_controls_its_own_neighbourhood():
    xs, ys = make_grid(53, 34)
    pa, _ = pitch_control_xy(_still((0, 0)), _still((30, 0)), (0, 0), xs, ys)
    assert _at(pa, xs, ys, 1, 0) > 0.95
    assert _at(pa, xs, ys, 29, 0) < 0.05


def test_symmetric_players_split_the_midpoint():
    xs, ys = make_grid(53, 34)
    pa, _ = pitch_control_xy(_still((-10, 0)), _still((10, 0)), (0, 0), xs, ys)
    assert abs(_at(pa, xs, ys, 0, 0) - 0.5) < 0.05


def test_running_toward_a_spot_helps():
    xs, ys = make_grid(53, 34)
    still = np.array([[-10, 0, 0.0, 0.0]])
    running = np.array([[-10, 0, 5.0, 0.0]])
    defender = _still((10, 0))
    p_still, _ = pitch_control_xy(still, defender, (0, 0), xs, ys)
    p_run, _ = pitch_control_xy(running, defender, (0, 0), xs, ys)
    assert _at(p_run, xs, ys, 2, 0) > _at(p_still, xs, ys, 2, 0)