import numpy as np
import pandas as pd

from pitchxt.dfl import _convert_tracking, _shots_to_frames


def _raw():
    return pd.DataFrame({
        "frame_id": [10, 11, 12, 100, 101],
        "period_id": [1, 1, 1, 2, 2],
        "timestamp": pd.to_timedelta([0, 0.04, 0.08, 0, 0.04], unit="s"),
        "ball_state": ["alive", "alive", "dead", "alive", "alive"],
        "ball_x": [0.5, 0.6, 0.5, 0.5, 0.5],
        "ball_y": [0.5, 0.5, 0.25, 0.5, 0.5],
        "P1_x": [0.0, 0.1, 0.2, np.nan, np.nan],
        "P1_y": [0.5, 0.5, 0.5, np.nan, np.nan],
    })


def test_convert_tracking_units_axes_and_index():
    out = _convert_tracking(_raw(), {"P1": "home_7", "P2": "away_9"}, 100.0, 50.0)
    assert out.index.name == "Frame" and out.index.tolist() == [10, 11, 12, 100, 101]
    assert np.allclose(out["ball_x"], [0, 10, 0, 0, 0])
    assert np.allclose(out["ball_y"], [0, 0, 12.5, 0, 0])      # y flipped: top of the pitch is +y
    assert np.allclose(out["home_7_x"][:3], [-50, -40, -30])   # x = 0 is the left goal line
    assert out["ball_alive"].tolist() == [True, True, False, True, True]
    assert np.allclose(out["time"], [0, 0.04, 0.08, 0, 0.04])
    assert "away_9_x" not in out.columns                        # player with no tracking data is skipped


def test_shots_are_placed_on_the_right_frame_per_period():
    tracking = _convert_tracking(_raw(), {}, 100.0, 50.0)
    rows = [("home", 1, 0.08, "SAVED", 1.0, 2.0), ("away", 2, 0.04, "GOAL", 3.0, 4.0)]
    shots = _shots_to_frames(rows, tracking)
    assert shots["start_frame"].tolist() == [12, 101]
    assert shots["goal"].tolist() == [False, True]