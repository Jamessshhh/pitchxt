"""Does threat share predict shots? Frame-level validation against Metrica events.

For sampled frames and for each team separately we ask: does this team take a
shot in the next `horizon_s` seconds? Each candidate score is judged by AUC
(0.5 = no signal, 1.0 = perfect).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .events import get_shots, load_metrica_events
from .tracking import PITCH_LENGTH, PITCH_WIDTH, add_velocities, load_tracking, to_single_direction
from .value import SB_L, SB_W, frame_value, voronoi_share
from .xt import xt_lookup

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
FPS = 25


def auc(score, label):
    """ROC AUC via the rank-sum formula (ties get average ranks)."""
    score = pd.Series(np.asarray(score, dtype=float))
    label = np.asarray(label, dtype=bool)
    n1 = int(label.sum())
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = score.rank().to_numpy()
    return (ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _ball_xt(xt_grid, bx, by, team):
    """xT at the ball's location, from `team`'s attacking direction."""
    if team == "away":
        bx, by = -bx, -by
    return xt_lookup(xt_grid, (bx + PITCH_LENGTH / 2) / PITCH_LENGTH * SB_L,
                     (PITCH_WIDTH / 2 - by) / PITCH_WIDTH * SB_W)


def frame_table(game_id, xt_grid, step=25, nx=33, ny=21, refresh=False):
    """Per-sampled-frame scores for both teams, cached to data/processed."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    path = PROC_DIR / f"frames_game{game_id}_step{step}_{nx}x{ny}_v2.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    df = to_single_direction(add_velocities(load_tracking(game_id)))
    rows = []
    for f in df.index[::step]:
        try:
            r = frame_value(df, f, xt_grid, nx, ny)
        except ValueError:          # ball not tracked at this frame
            continue
        bx, by = df.loc[f, "ball_x"], df.loc[f, "ball_y"]
        vor = voronoi_share(df, f, xt_grid, nx, ny)
        for team, share, v in (("home", r["threat_share"], vor), ("away", 1 - r["threat_share"], 1 - vor)):
            rows.append(dict(game=game_id, frame=f, period=df.loc[f, "period"], team=team,
                             share=share, share_voronoi=v, ball_xt=_ball_xt(xt_grid, bx, by, team)))
    out = pd.DataFrame(rows)
    out.to_parquet(path, index=False)
    return out


def label_frames(table, shots, horizon_s=10):
    """Add 'label': True if the team takes a shot in the next horizon_s seconds."""
    out = table.copy()
    h = horizon_s * FPS
    label = np.zeros(len(out), dtype=bool)
    for (game, team), idx in out.groupby(["game", "team"]).groups.items():
        s = shots[(shots["game"] == game) & (shots["team"] == team)]["start_frame"].to_numpy()
        f = out.loc[idx, "frame"].to_numpy()[:, None]
        label[out.index.get_indexer(idx)] = ((s[None, :] > f) & (s[None, :] <= f + h)).any(axis=1)
    out["label"] = label
    return out


def bootstrap_auc(table, col, n_boot=500, block_frames=1500, seed=0):
    """95% CI for AUC, resampling 60-second blocks (frames close in time are not independent)."""
    rng = np.random.default_rng(seed)
    block = table["game"].astype(int) * 10_000 + table["frame"] // block_frames
    groups = [g for _, g in table.groupby(block)]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in pick])
        vals.append(auc(sample[col], sample["label"]))
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def run_validation(game_ids, xt_grid, horizon_s=10, step=25, nx=33, ny=21):
    """Returns (labelled frame table, summary DataFrame of AUCs with CIs)."""
    tables, shots = [], []
    for g in game_ids:
        tables.append(frame_table(g, xt_grid, step, nx, ny))
        s = get_shots(load_metrica_events(g))
        s["game"] = g
        shots.append(s)
    table = label_frames(pd.concat(tables, ignore_index=True), pd.concat(shots, ignore_index=True), horizon_s)

    rows = []
    for col, name in (("ball_xt", "xT at ball location"),
                      ("share_voronoi", "threat share, Voronoi control x xT"),
                      ("share", "threat share, Spearman control x xT")):
        lo, hi = bootstrap_auc(table, col)
        rows.append((name, auc(table[col], table["label"]), lo, hi))
    summary = pd.DataFrame(rows, columns=["score", "auc", "ci_low", "ci_high"])
    return table, summary


def paired_bootstrap(table, col_a, col_b, n_boot=1000, block_frames=1500, seed=0):
    """AUC(col_a) - AUC(col_b) on the same resampled blocks: (difference, ci_low, ci_high, P(diff>0))."""
    rng = np.random.default_rng(seed)
    block = table["game"].astype(int) * 10_000 + table["frame"] // block_frames
    groups = [g for _, g in table.groupby(block)]
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in pick])
        diffs.append(auc(sample[col_a], sample["label"]) - auc(sample[col_b], sample["label"]))
    diffs = np.array(diffs)
    point = auc(table[col_a], table["label"]) - auc(table[col_b], table["label"])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return point, lo, hi, float(np.nanmean(diffs > 0))


def paired_summary(table):
    """The comparisons that matter: does each richer model beat the simpler one?"""
    pairs = [
        ("share", "ball_xt", "Spearman share vs xT at ball"),
        ("share", "share_voronoi", "Spearman share vs Voronoi share"),
        ("share_voronoi", "ball_xt", "Voronoi share vs xT at ball"),
    ]
    rows = [(name, *paired_bootstrap(table, a, b)) for a, b, name in pairs]
    return pd.DataFrame(rows, columns=["comparison", "auc_diff", "ci_low", "ci_high", "p_better"])