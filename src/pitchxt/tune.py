"""Tune the Spearman pitch-control parameters with match-level cross-validation.

Random search over a few parameters. Every candidate is scored once on every match
(AUC of threat share for "shot within horizon"). Then, for each fold of held-out
matches, the best candidate on the *other* matches is chosen and judged on the
held-out ones, so the reported improvement never uses the matches it is tested on.
"""
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .matches import load_match, load_shots
from .pitch_control import DEFAULT_PARAMS, make_grid, pitch_control_xy
from .tracking import frame_players
from .validate import FPS, auc
from .value import xt_on_control_grid

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"

# name: (low, high, sampled on a log scale?)
SEARCH_SPACE = {
    "reaction_time": (0.3, 1.2, False),
    "max_player_speed": (4.0, 8.0, False),
    "tti_sigma": (0.2, 0.9, False),
    "lambda_att": (1.5, 12.0, True),
    "average_ball_speed": (10.0, 25.0, False),
}


def sample_trials(n, seed=0):
    """Trial 0 is the published defaults; the rest are random draws from SEARCH_SPACE."""
    rng = np.random.default_rng(seed)
    trials = [{k: DEFAULT_PARAMS[k] for k in SEARCH_SPACE}]
    for _ in range(n - 1):
        t = {}
        for k, (lo, hi, log) in SEARCH_SPACE.items():
            t[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if log else float(rng.uniform(lo, hi))
        trials.append(t)
    return trials


def build_cache(game_id, step=75, horizon_s=10, refresh=False):
    """Pre-extract sampled frames (player arrays, ball, labels) so trials don't touch pandas."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    path = PROC_DIR / f"tune_{game_id}_step{step}_h{horizon_s}.pkl"
    if path.exists() and not refresh:
        with open(path, "rb") as fh:
            return pickle.load(fh)

    df, _ = load_match(game_id)
    shots = load_shots(game_id)
    alive = df["ball_alive"].to_numpy()
    frames, home, away, ball = [], [], [], []
    for pos in range(0, len(df), step):
        if not alive[pos]:
            continue
        f = df.index[pos]
        b = (df.at[f, "ball_x"], df.at[f, "ball_y"])
        if np.isnan(b).any():
            continue
        frames.append(f)
        home.append(frame_players(df, f, "home"))
        away.append(frame_players(df, f, "away"))
        ball.append(b)
    frames = np.array(frames)
    h = horizon_s * FPS

    def label(team):
        s = shots.loc[shots["team"] == team, "start_frame"].to_numpy()
        return ((s[None, :] > frames[:, None]) & (s[None, :] <= frames[:, None] + h)).any(axis=1)

    cache = dict(game=str(game_id), frames=frames, home=home, away=away, ball=np.array(ball),
                 y_home=label("home"), y_away=label("away"))
    with open(path, "wb") as fh:
        pickle.dump(cache, fh)
    return cache


_G = {}


def _init(caches, xt_grid, nx, ny):
    xs, ys = make_grid(nx, ny)
    _G.update(caches=caches, xs=xs, ys=ys,
              xt_h=xt_on_control_grid(xt_grid, xs, ys, "home"),
              xt_a=xt_on_control_grid(xt_grid, xs, ys, "away"))


def _score_trial(params):
    """Home threat share for every cached frame of every match, under one parameter set."""
    out = []
    for c in _G["caches"]:
        share = np.empty(len(c["frames"]))
        for i in range(len(share)):
            pc_h, pc_a = pitch_control_xy(c["home"][i], c["away"][i], c["ball"][i], _G["xs"], _G["ys"], params)
            vh = (pc_h * _G["xt_h"]).mean()
            va = (pc_a * _G["xt_a"]).mean()
            share[i] = vh / (vh + va) if (vh + va) > 0 else 0.5
        out.append(share)
    return out


def run_search(caches, xt_grid, trials, nx=25, ny=16, workers=None):
    """trial_shares[t][m] = home-share array of match m under trial t."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    results = [None] * len(trials)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(caches, xt_grid, nx, ny)) as ex:
        futures = {ex.submit(_score_trial, t): i for i, t in enumerate(trials)}
        done = 0
        for fut in futures:
            results[futures[fut]] = fut.result()
            done += 1
            print(f"  trial {done}/{len(trials)} done", flush=True)
    return results


def _match_arrays(cache, share):
    """Scores, labels and frame ids for both teams of one match."""
    scores = np.concatenate([share, 1 - share])
    labels = np.concatenate([cache["y_home"], cache["y_away"]])
    frames = np.concatenate([cache["frames"], cache["frames"]])
    return scores, labels, frames


def pooled_auc(shares_per_match, caches, match_ids):
    s, y = [], []
    for m in match_ids:
        sc, lb, _ = _match_arrays(caches[m], shares_per_match[m])
        s.append(sc)
        y.append(lb)
    return auc(np.concatenate(s), np.concatenate(y))


def paired_auc_diff(score_a, score_b, label, block, n_boot=1000, seed=0):
    """AUC(a) - AUC(b) with a block bootstrap: (difference, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(block, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(block[order])) + 1)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        diffs.append(auc(score_a[idx], label[idx]) - auc(score_b[idx], label[idx]))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return auc(score_a, label) - auc(score_b, label), lo, hi


def cross_validate(trial_shares, caches, k=3, seed=0, n_boot=1000):
    """Leave-matches-out search. Trial 0 must be the default parameters.

    Returns (per-fold DataFrame, dict with the pooled held-out comparison)."""
    n_m = len(caches)
    k = min(k, n_m)
    order = np.random.default_rng(seed).permutation(n_m)
    folds = [sorted(order[i::k].tolist()) for i in range(k)]
    rows = []
    held = dict(default=[], tuned=[], label=[], block=[])
    for fi, test in enumerate(folds):
        train = [m for m in range(n_m) if m not in test]
        train_auc = [pooled_auc(ts, caches, train) for ts in trial_shares]
        best = int(np.nanargmax(train_auc))
        rows.append(dict(fold=fi, test_matches=",".join(caches[m]["game"] for m in test), best_trial=best,
                         train_default=train_auc[0], train_best=train_auc[best],
                         test_default=pooled_auc(trial_shares[0], caches, test),
                         test_tuned=pooled_auc(trial_shares[best], caches, test)))
        for m in test:
            sd, lb, fr = _match_arrays(caches[m], trial_shares[0][m])
            st, _, _ = _match_arrays(caches[m], trial_shares[best][m])
            held["default"].append(sd)
            held["tuned"].append(st)
            held["label"].append(lb)
            held["block"].append(m * 1_000_000 + fr // (60 * FPS))
    held = {k_: np.concatenate(v) for k_, v in held.items()}
    diff, lo, hi = paired_auc_diff(held["tuned"], held["default"], held["label"], held["block"], n_boot)
    summary = dict(auc_default=auc(held["default"], held["label"]), auc_tuned=auc(held["tuned"], held["label"]),
                   diff=diff, ci_low=lo, ci_high=hi)
    return pd.DataFrame(rows), summary



def grid_trials_for(axes):
    """axes: {parameter name: list of values}. Trial 0 is the defaults; then every combination
    (first axis varies slowest) with all other parameters at their defaults."""
    import itertools

    base = {k: DEFAULT_PARAMS[k] for k in SEARCH_SPACE}
    trials = [dict(base)]
    for combo in itertools.product(*axes.values()):
        trials.append({**base, **{k: float(v) for k, v in zip(axes.keys(), combo)}})
    return trials


def grid_trials(reaction_times, ball_speeds):
    """Reaction time x ball speed grid (see grid_trials_for)."""
    return grid_trials_for({"reaction_time": reaction_times, "average_ball_speed": ball_speeds})



BASELINE_NAMES = (
    "xT at ball",
    "attacking depth (own - opponent)",
    "forward speed (own)",
    "forward speed (own - opponent)",
)


def baseline_scores(cache, xt_grid):
    """Simple scores per cached frame and team, with no pitch control at all.

    Returns {name: (home_rows, away_rows)}. Home attacks toward +x and away toward -x.
    'depth' is the mean x of a team's players, 'forward speed' their mean velocity toward goal."""
    from .validate import _ball_xt

    n = len(cache["frames"])
    out = {k: (np.zeros(n), np.zeros(n)) for k in BASELINE_NAMES}
    mean = lambda arr, j: float(arr[:, j].mean()) if len(arr) else 0.0
    for i in range(n):
        hx, hv = mean(cache["home"][i], 0), mean(cache["home"][i], 2)
        ax, av = mean(cache["away"][i], 0), mean(cache["away"][i], 2)
        depth_h, depth_a = hx, -ax
        fs_h, fs_a = hv, -av
        bx, by = cache["ball"][i]
        vals = {
            "xT at ball": (_ball_xt(xt_grid, bx, by, "home"), _ball_xt(xt_grid, bx, by, "away")),
            "attacking depth (own - opponent)": (depth_h - depth_a, depth_a - depth_h),
            "forward speed (own)": (fs_h, fs_a),
            "forward speed (own - opponent)": (fs_h - fs_a, fs_a - fs_h),
        }
        for k, (h_val, a_val) in vals.items():
            out[k][0][i], out[k][1][i] = h_val, a_val
    return out


def stack_rows(caches, home_rows, away_rows):
    """Concatenate per-match home and away rows into (scores, labels, 60-second block ids)."""
    s, y, b = [], [], []
    for m, c in enumerate(caches):
        s.append(np.concatenate([home_rows[m], away_rows[m]]))
        y.append(np.concatenate([c["y_home"], c["y_away"]]))
        fr = np.concatenate([c["frames"], c["frames"]])
        b.append(m * 1_000_000 + fr // (60 * FPS))
    return np.concatenate(s), np.concatenate(y), np.concatenate(b)