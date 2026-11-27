"""Does threat share predict shots? Frame-level validation against match events.

For sampled frames and for each team separately we ask: does this team take a
shot in the next `horizon_s` seconds? Each candidate score is judged by AUC
(0.5 = no signal, 1.0 = perfect). Works on any mix of Metrica games (integer ids)
and DFL matches (string ids such as 'J03WPY').
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .matches import load_match, load_shots, source_of
from .tracking import PITCH_LENGTH, PITCH_WIDTH, frame_players
from .value import SB_L, SB_W, frame_value, voronoi_share
from .xt import xt_lookup

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
FPS = 25
# parameters chosen by match-level cross-validation (see tune.py): 5 s look-ahead, 0.3 s reaction delay
TUNED_PARAMS = {"lookahead": 5.0, "reaction_time": 0.3}

ALL_SCORES = (
    ("ball_xt", "xT at ball location"),
    ("depth", "attacking depth (own - opponent)"),
    ("share_voronoi", "threat share, Voronoi control x xT"),
    ("share", "threat share, Spearman control x xT"),
    ("share_tuned", "threat share, Spearman + 5 s look-ahead"),
    ("gnn", "graph attention network (learned)"),
)


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
    """Per-sampled-frame scores for both teams (ball-in-play frames only), cached to data/processed."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    path = PROC_DIR / f"frames_game{game_id}_step{step}_{nx}x{ny}_v2.parquet"
    if path.exists() and not refresh:
        out = pd.read_parquet(path)
        out["game"] = out["game"].astype(str)
        return out

    df, _ = load_match(game_id)
    alive = df["ball_alive"].to_numpy()
    rows = []
    for pos in range(0, len(df), step):
        if not alive[pos]:
            continue
        f = df.index[pos]
        try:
            r = frame_value(df, f, xt_grid, nx, ny)
        except ValueError:          # ball not tracked at this frame
            continue
        bx, by = df.loc[f, "ball_x"], df.loc[f, "ball_y"]
        vor = voronoi_share(df, f, xt_grid, nx, ny)
        for team, share, v in (("home", r["threat_share"], vor), ("away", 1 - r["threat_share"], 1 - vor)):
            rows.append(dict(game=str(game_id), frame=f, period=df.loc[f, "period"], team=team,
                             share=share, share_voronoi=v, ball_xt=_ball_xt(xt_grid, bx, by, team)))
    out = pd.DataFrame(rows)
    out.to_parquet(path, index=False)
    return out


def _tuned_path(game_id, params, step, nx, ny):
    tag = "_".join(f"{k}{v:g}" for k, v in sorted(params.items()))
    return PROC_DIR / f"frames_tuned_game{game_id}_step{step}_{nx}x{ny}_{tag}.parquet"


def tuned_table(game_id, xt_grid, params, step=25, nx=33, ny=21, refresh=False):
    """Threat share under custom Spearman parameters, plus the team-depth baseline, for the same
    sampled frames as frame_table. Cached to data/processed."""
    path = _tuned_path(game_id, params, step, nx, ny)
    if path.exists() and not refresh:
        out = pd.read_parquet(path)
        out["game"] = out["game"].astype(str)
        return out

    df, _ = load_match(game_id)
    alive = df["ball_alive"].to_numpy()
    rows = []
    for pos in range(0, len(df), step):
        if not alive[pos]:
            continue
        f = df.index[pos]
        try:
            r = frame_value(df, f, xt_grid, nx, ny, params)
        except ValueError:
            continue
        home, away = frame_players(df, f, "home"), frame_players(df, f, "away")
        hx = float(home[:, 0].mean()) if len(home) else 0.0
        ax = float(away[:, 0].mean()) if len(away) else 0.0
        d = hx + ax          # home depth (mean x) minus away depth (mean of -x)
        for team, share, depth in (("home", r["threat_share"], d), ("away", 1 - r["threat_share"], -d)):
            rows.append(dict(game=str(game_id), frame=f, team=team, share_tuned=share, depth=depth))
    out = pd.DataFrame(rows)
    out.to_parquet(path, index=False)
    return out


def _tuned_worker(args):
    game_id, xt_grid, params, step, nx, ny = args
    tuned_table(game_id, xt_grid, params, step, nx, ny)
    return str(game_id)


def precompute_tuned(game_ids, xt_grid, params, step=25, nx=33, ny=21, workers=4):
    """Fill the tuned-table cache for every match in parallel (call from under `if __name__ == "__main__"`)."""
    todo = [g for g in game_ids if not _tuned_path(g, params, step, nx, ny).exists()]
    if not todo:
        return
    print(f"computing tuned scores for {len(todo)} matches with {workers} workers...", flush=True)
    jobs = [(g, xt_grid, params, step, nx, ny) for g in todo]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for done in ex.map(_tuned_worker, jobs):
            print(f"  finished {done}", flush=True)


def label_frames(table, shots, horizon_s=10, fps=FPS):
    """Add 'label': True if the team takes a shot in the next horizon_s seconds."""
    out = table.copy()
    h = int(horizon_s * fps)
    label = np.zeros(len(out), dtype=bool)
    for (game, team), idx in out.groupby(["game", "team"]).groups.items():
        s = shots[(shots["game"] == game) & (shots["team"] == team)]["start_frame"].to_numpy()
        f = out.loc[idx, "frame"].to_numpy()[:, None]
        label[out.index.get_indexer(idx)] = ((s[None, :] > f) & (s[None, :] <= f + h)).any(axis=1)
    out["label"] = label
    return out


def _blocks(table, block_frames):
    """Block id per row: game x 60-second window (frames close in time are not independent)."""
    return pd.factorize(table["game"])[0] * 1_000_000 + table["frame"].to_numpy() // block_frames


def bootstrap_auc(table, col, n_boot=500, block_frames=1500, seed=0):
    """95% CI for AUC, resampling 60-second blocks."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in table.groupby(_blocks(table, block_frames))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in pick])
        vals.append(auc(sample[col], sample["label"]))
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def _summarise(table, block_frames=1500):
    rows = []
    for col, name in ALL_SCORES:
        if col not in table.columns:
            continue
        lo, hi = bootstrap_auc(table, col, block_frames=block_frames)
        rows.append((name, auc(table[col], table["label"]), lo, hi))
    return pd.DataFrame(rows, columns=["score", "auc", "ci_low", "ci_high"])


def run_validation(game_ids, xt_grid, horizon_s=10, step=25, nx=33, ny=21, tuned=None):
    """Returns (labelled frame table, summary DataFrame of AUCs with CIs).
    Pass tuned=TUNED_PARAMS to add the look-ahead model and the team-depth baseline."""
    tables, shots = [], []
    for g in game_ids:
        t = frame_table(g, xt_grid, step, nx, ny)
        if tuned is not None:
            t = t.merge(tuned_table(g, xt_grid, tuned, step, nx, ny), on=["game", "frame", "team"], how="inner")
        tables.append(t)
        s = load_shots(g)[["team", "period", "start_frame", "goal"]].copy()
        s["game"] = str(g)
        shots.append(s)
    table = label_frames(pd.concat(tables, ignore_index=True), pd.concat(shots, ignore_index=True), horizon_s)
    table["source"] = table["game"].map(source_of)
    return table, _summarise(table)


def summary_by_source(table):
    """The same AUC table computed separately for each data source."""
    parts = []
    for src, sub in table.groupby("source"):
        s = _summarise(sub)
        s.insert(0, "source", src)
        s.insert(1, "positive_frames", int(sub["label"].sum()))
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def paired_bootstrap(table, col_a, col_b, n_boot=1000, block_frames=1500, seed=0):
    """AUC(col_a) - AUC(col_b) on the same resampled blocks: (difference, ci_low, ci_high, P(diff>0))."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in table.groupby(_blocks(table, block_frames))]
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in pick])
        diffs.append(auc(sample[col_a], sample["label"]) - auc(sample[col_b], sample["label"]))
    diffs = np.array(diffs)
    point = auc(table[col_a], table["label"]) - auc(table[col_b], table["label"])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return point, lo, hi, float(np.nanmean(diffs > 0))


def paired_summary(table, block_frames=1500):
    """The comparisons that matter: does each richer model beat the simpler one?"""
    pairs = [
        ("share", "ball_xt", "Spearman share vs xT at ball"),
        ("share", "share_voronoi", "Spearman share vs Voronoi share"),
        ("share_voronoi", "ball_xt", "Voronoi share vs xT at ball"),
    ]
    if "share_tuned" in table.columns:
        pairs += [
            ("share_tuned", "depth", "Tuned Spearman vs team depth"),
            ("share_tuned", "share", "Tuned Spearman vs published Spearman"),
            ("share", "depth", "Published Spearman vs team depth"),
            ("share_tuned", "ball_xt", "Tuned Spearman vs xT at ball"),
        ]
    if "gnn" in table.columns:
        pairs += [
            ("gnn", "share_tuned", "Graph network vs Spearman + look-ahead"),
            ("gnn", "share", "Graph network vs published Spearman"),
            ("gnn", "depth", "Graph network vs team depth"),
        ]
    rows = [(name, *paired_bootstrap(table, a, b, block_frames=block_frames)) for a, b, name in pairs]
    return pd.DataFrame(rows, columns=["comparison", "auc_diff", "ci_low", "ci_high", "p_better"])