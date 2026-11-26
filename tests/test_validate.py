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



def test_paired_bootstrap_detects_a_better_score():
    from pitchxt.validate import paired_bootstrap

    rng = np.random.default_rng(1)
    n = 3000
    label = rng.random(n) < 0.2
    table = pd.DataFrame({
        "game": 1, "frame": np.arange(n) * 25, "label": label,
        "good": label * 1.0 + rng.normal(0, 0.8, n),     # informative
        "noise": rng.normal(0, 1, n),                    # pure noise
    })
    diff, lo, hi, p = paired_bootstrap(table, "good", "noise", n_boot=200)
    assert diff > 0.2 and lo > 0 and p > 0.99
    same, *_ = paired_bootstrap(table, "good", "good", n_boot=50)
    assert same == 0



def test_source_of_and_block_ids_handle_mixed_game_ids():
    from pitchxt.matches import source_of
    from pitchxt.validate import _blocks

    assert source_of(1) == "metrica" and source_of("2") == "metrica"
    assert source_of("J03WPY") == "dfl"
    table = pd.DataFrame({"game": ["1", "1", "J03WPY", "J03WPY"], "frame": [0, 2000, 0, 2000]})
    assert len(set(_blocks(table, 1500))) == 4        # same frame in different games is a different block



def test_summary_and_paired_comparisons_include_the_tuned_columns_when_present():
    from pitchxt.validate import _summarise, paired_summary

    rng = np.random.default_rng(2)
    n = 3000
    label = rng.random(n) < 0.15
    table = pd.DataFrame({"game": "1", "frame": np.arange(n) * 25, "label": label})
    for col, signal in (("ball_xt", 0.3), ("depth", 0.4), ("share_voronoi", 0.4), ("share", 0.5), ("share_tuned", 0.9)):
        table[col] = signal * label + rng.normal(0, 1, n)
    assert len(_summarise(table)) == 5
    paired = paired_summary(table)
    assert "Tuned Spearman vs team depth" in set(paired["comparison"])
    row = paired[paired["comparison"] == "Tuned Spearman vs team depth"].iloc[0]
    assert row["auc_diff"] > 0.1 and row["ci_low"] > 0