import numpy as np
import pandas as pd

from pitchxt.gnn_data import (HORIZONS, MAX_PER_TEAM, _ball_arrays, _team_arrays, build_from_tracking,
                              sample_positions, shot_labels)


def _tracking(n=60, alive_until=None):
    """Two home and one away player, home_2 absent for the first 10 rows, ball moving +x at 5 m/s (25 fps)."""
    idx = pd.Index(np.arange(1000, 1000 + n), name="Frame")
    df = pd.DataFrame(index=idx)
    t = np.arange(n) / 25
    df["period"] = 1
    df["ball_x"], df["ball_y"] = 5.0 * t, 0.0
    df["ball_alive"] = True if alive_until is None else (np.arange(n) < alive_until)
    for name, x0, y0 in (("home_1", -10.0, 3.0), ("home_2", -5.0, -3.0), ("away_9", 20.0, 0.0)):
        df[f"{name}_x"], df[f"{name}_y"] = x0, y0
        df[f"{name}_vx"], df[f"{name}_vy"] = 1.0, -0.5
    df.loc[df.index[:10], ["home_2_x", "home_2_y"]] = np.nan
    return df


def test_sampling_skips_dead_ball_and_missing_ball_frames():
    df = _tracking(60, alive_until=40)
    df.loc[df.index[25], "ball_x"] = np.nan
    pos = sample_positions(df, step=5)
    assert pos.tolist() == [0, 5, 10, 15, 20, 30, 35]          # 25 has no ball, 40+ is dead ball


def test_team_arrays_put_real_players_first_and_mask_the_rest():
    df = _tracking()
    arr, mask = _team_arrays(df, "home", np.array([0, 20]))
    assert arr.shape == (2, MAX_PER_TEAM, 4) and mask.shape == (2, MAX_PER_TEAM)
    assert mask[0].sum() == 1 and mask[1].sum() == 2            # home_2 is missing at row 0
    assert arr[0, 0, :2].tolist() == [-10.0, 3.0]
    assert (arr[~mask] == 0).all()                              # padding is zeros
    assert sorted(arr[1, :2, 0].tolist()) == [-10.0, -5.0]


def test_ball_velocity_comes_from_a_short_centred_difference():
    df = _tracking()
    ball = _ball_arrays(df, np.array([20]), fps=25)
    assert np.allclose(ball[0], [5.0 * 20 / 25, 0.0, 5.0, 0.0], atol=1e-4)
    edge = _ball_arrays(df, np.array([0]), fps=25)               # one-sided at the start of the table, still sensible
    assert np.isclose(edge[0, 2], 5.0, atol=1e-4)


def test_shot_labels_look_forward_only():
    frames = np.array([100, 200, 300, 400])
    assert shot_labels(frames, [310], 100).tolist() == [False, False, True, False]
    assert shot_labels(frames, [], 100).tolist() == [False] * 4


def test_build_from_tracking_returns_aligned_arrays():
    df = _tracking(100)
    shots = pd.DataFrame({"team": ["home", "away"], "start_frame": [1060, 1090]})
    d = build_from_tracking(df, shots, fps=25, step_s=0.4)      # every 10th row
    n = len(d["frames"])
    assert n == 10 and d["players"].shape == (n, 2, MAX_PER_TEAM, 4) and d["mask"].shape == (n, 2, MAX_PER_TEAM)
    assert d["ball"].shape == (n, 4) and d["n_detected"].tolist() == [22] * n
    h = int(5 * 25)
    assert d["y_home_5"][d["frames"] == 1000][0]                 # shot at 1060 is within 5 s (125 frames) of 1000
    assert not d["y_away_5"][d["frames"] == 1090][0]             # a shot at the same frame is not "in the future"
    assert set(f"y_home_{h_}" for h_ in HORIZONS) <= set(d)