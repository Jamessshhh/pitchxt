import numpy as np
import pandas as pd

from pitchxt.validate import auc, label_frames


def test_auc_perfect_and_inverted():
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0


def test_auc_ties_and_random():
    assert auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    rng = np.random.default_rng(0)
    assert abs(auc(rng.random(5000), rng.random(5000) < 0.3) - 0.5) < 0.03


def test_label_frames_looks_forward_only():
    table = pd.DataFrame({"game": [1] * 4, "team": ["home"] * 4, "frame": [100, 200, 300, 400]})
    shots = pd.DataFrame({"game": [1], "team": ["home"], "start_frame": [310]})
    out = label_frames(table, shots, horizon_s=4)      # 4 s = 100 frames
    assert out["label"].tolist() == [False, False, True, False]


def test_direction_is_inferred_per_game():
    from pitchxt.tracking import home_attacks_right_first_half, to_single_direction

    def frames(gk_x):
        return pd.DataFrame({
            "period": [1, 1, 2, 2],
            "home_1_x": [gk_x, gk_x, -gk_x, -gk_x], "home_1_y": [0.0] * 4,
            "home_2_x": [5.0, 5.0, -5.0, -5.0], "home_2_y": [0.0] * 4,
        })

    left_keeper, right_keeper = frames(-40.0), frames(40.0)
    assert home_attacks_right_first_half(left_keeper)
    assert not home_attacks_right_first_half(right_keeper)
    # after normalising, the keeper is always on the left (home attacks +x) in both halves
    for df in (left_keeper, right_keeper):
        assert (to_single_direction(df)["home_1_x"] < 0).all()