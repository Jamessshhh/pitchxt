import numpy as np

from pitchxt.tune import cross_validate, paired_auc_diff, pooled_auc, sample_trials


def _caches(n_matches=6, n=400, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for m in range(n_matches):
        y = rng.random(n) < 0.15
        out.append(dict(game=str(m), frames=np.arange(n) * 75, y_home=y, y_away=rng.random(n) < 0.15))
    return out


def _shares(caches, signal, seed):
    """Home-share arrays whose informativeness is controlled by `signal`."""
    rng = np.random.default_rng(seed)
    return [np.clip(0.5 + signal * (c["y_home"] * 1.0 - 0.15) + rng.normal(0, 0.2, len(c["frames"])), 0, 1)
            for c in caches]


def test_sample_trials_start_with_the_defaults_and_stay_in_range():
    from pitchxt.pitch_control import DEFAULT_PARAMS
    from pitchxt.tune import SEARCH_SPACE

    trials = sample_trials(25, seed=3)
    assert len(trials) == 25 and trials[0] == {k: DEFAULT_PARAMS[k] for k in SEARCH_SPACE}
    for t in trials[1:]:
        for k, (lo, hi, _) in SEARCH_SPACE.items():
            assert lo <= t[k] <= hi


def test_cross_validation_picks_the_informative_trial_and_beats_default():
    caches = _caches()
    trial_shares = [_shares(caches, 0.0, 1), _shares(caches, 0.6, 2), _shares(caches, 0.1, 3)]
    folds, overall = cross_validate(trial_shares, caches, k=3, n_boot=100)
    assert (folds["best_trial"] == 1).all()
    assert overall["auc_tuned"] > overall["auc_default"] + 0.1
    assert overall["ci_low"] > 0


def test_tuning_noise_gives_no_confident_gain():
    caches = _caches(seed=5)
    trial_shares = [_shares(caches, 0.0, s) for s in range(10, 16)]    # all pure noise
    _, overall = cross_validate(trial_shares, caches, k=3, n_boot=100)
    assert overall["ci_low"] <= 0 <= overall["ci_high"] or abs(overall["diff"]) < 0.03


def test_paired_diff_of_identical_scores_is_zero():
    rng = np.random.default_rng(0)
    s, y = rng.random(500), rng.random(500) < 0.3
    d, lo, hi = paired_auc_diff(s, s, y, np.arange(500) // 50, n_boot=50)
    assert d == 0 and lo == 0 and hi == 0
    assert pooled_auc is not None



def test_grid_trials_cover_the_grid_and_keep_other_parameters_at_defaults():
    from pitchxt.pitch_control import DEFAULT_PARAMS
    from pitchxt.tune import grid_trials

    trials = grid_trials([0.5, 1.0, 2.0], [15, 30])
    assert len(trials) == 1 + 3 * 2
    assert trials[0]["reaction_time"] == DEFAULT_PARAMS["reaction_time"]
    for t in trials[1:]:
        assert t["tti_sigma"] == DEFAULT_PARAMS["tti_sigma"] and t["lambda_att"] == DEFAULT_PARAMS["lambda_att"]
    assert {(t["reaction_time"], t["average_ball_speed"]) for t in trials[1:]} == {
        (r, b) for r in (0.5, 1.0, 2.0) for b in (15.0, 30.0)}
    


def test_baseline_scores_read_depth_and_forward_speed_from_each_teams_own_view():
    from pitchxt.tune import baseline_scores

    xt = np.ones((12, 16)) * 0.01
    # home players at x = +10 moving +x at 4 m/s; away players at x = +30 moving -x at 3 m/s
    cache = dict(frames=np.array([100]), ball=np.array([[0.0, 0.0]]),
                 home=[np.array([[10.0, 0.0, 4.0, 0.0], [10.0, 5.0, 4.0, 0.0]])],
                 away=[np.array([[30.0, 0.0, -3.0, 0.0]])])
    out = baseline_scores(cache, xt)
    assert out["forward speed (own)"][0][0] == 4.0 and out["forward speed (own)"][1][0] == 3.0   # both are moving toward goal
    assert out["forward speed (own - opponent)"][0][0] == 1.0 and out["forward speed (own - opponent)"][1][0] == -1.0
    assert out["attacking depth (own - opponent)"][0][0] == 10.0 - (-30.0)      # home 10 m advanced, away has retreated to -30 in its own frame
    assert out["xT at ball"][0][0] == 0.01