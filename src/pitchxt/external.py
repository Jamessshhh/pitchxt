"""External test: score the frozen models on matches that were never used to tune anything.

Nothing is fitted here. The Spearman parameters are the published ones and the tuned ones
(TUNED_PARAMS), exactly as selected on the Metrica and DFL matches. Works on the SkillCorner
open data (10 fps broadcast tracking), so shot windows and bootstrap blocks are in 10 fps frames.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .skillcorner import FPS, SC_DIR, load_skillcorner_match, load_skillcorner_shots
from .tracking import add_velocities, frame_players, to_single_direction
from .validate import _ball_xt, _summarise, auc, label_frames, paired_bootstrap, paired_summary
from .value import frame_value, voronoi_share

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
BLOCK_FRAMES = 60 * FPS          # 60-second bootstrap blocks at 10 fps


def _ext_path(match_id, params, step_s, nx, ny):
    tag = "_".join(f"{k}{v:g}" for k, v in sorted(params.items()))
    return PROC_DIR / f"ext_sc{match_id}_step{step_s:g}s_{nx}x{ny}_{tag}.parquet"


def external_table(match_id, xt_grid, params, step_s=1.0, nx=33, ny=21, data_dir=SC_DIR, refresh=False):
    """All five scores for both teams at one frame per `step_s` seconds of ball-in-play, cached."""
    path = _ext_path(match_id, params, step_s, nx, ny)
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    raw, _ = load_skillcorner_match(match_id, data_dir)
    df = to_single_direction(add_velocities(raw))
    step = max(1, int(round(FPS * step_s)))
    alive, n_det = df["ball_alive"].to_numpy(), df["n_detected"].to_numpy()
    rows = []
    for pos in range(0, len(df), step):
        if not alive[pos]:
            continue
        f = df.index[pos]
        try:
            r_pub = frame_value(df, f, xt_grid, nx, ny)
            r_tun = frame_value(df, f, xt_grid, nx, ny, params)
        except ValueError:                       # ball position missing
            continue
        vor = voronoi_share(df, f, xt_grid, nx, ny)
        bx, by = df.at[f, "ball_x"], df.at[f, "ball_y"]
        home, away = frame_players(df, f, "home"), frame_players(df, f, "away")
        d = (float(home[:, 0].mean()) if len(home) else 0.0) + (float(away[:, 0].mean()) if len(away) else 0.0)
        for team, sp, st, v, depth in (
                ("home", r_pub["threat_share"], r_tun["threat_share"], vor, d),
                ("away", 1 - r_pub["threat_share"], 1 - r_tun["threat_share"], 1 - vor, -d)):
            rows.append(dict(game=str(match_id), frame=f, period=df.at[f, "period"], team=team,
                             share=sp, share_tuned=st, share_voronoi=v, depth=depth,
                             ball_xt=_ball_xt(xt_grid, bx, by, team), n_detected=int(n_det[pos])))
    out = pd.DataFrame(rows)
    out.to_parquet(path, index=False)
    return out


def _worker(args):
    match_id, xt_grid, params, step_s, nx, ny, data_dir = args
    external_table(match_id, xt_grid, params, step_s, nx, ny, data_dir)
    return str(match_id)


def precompute_external(match_ids, xt_grid, params, step_s=1.0, nx=33, ny=21, data_dir=SC_DIR, workers=4):
    """Fill the cache for every match in parallel (call from under `if __name__ == "__main__"`)."""
    todo = [m for m in match_ids if not _ext_path(m, params, step_s, nx, ny).exists()]
    if not todo:
        return
    print(f"scoring {len(todo)} matches with {workers} workers (each parses ~90 MB of tracking)...", flush=True)
    jobs = [(m, xt_grid, params, step_s, nx, ny, data_dir) for m in todo]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for done in ex.map(_worker, jobs):
            print(f"  finished {done}", flush=True)


def external_labelled_table(match_ids, xt_grid, params, horizon_s=10, step_s=1.0, nx=33, ny=21,
                            data_dir=SC_DIR, min_detected=0):
    """The cached score tables with the 'shot within horizon_s' label attached (no bootstrap)."""
    tables, shots = [], []
    for m in match_ids:
        tables.append(external_table(m, xt_grid, params, step_s, nx, ny, data_dir))
        s = load_skillcorner_shots(m, data_dir)[["team", "period", "start_frame", "goal"]].copy()
        s["game"] = str(m)
        shots.append(s)
    table = label_frames(pd.concat(tables, ignore_index=True), pd.concat(shots, ignore_index=True), horizon_s, fps=FPS)
    return table[table["n_detected"] >= min_detected].reset_index(drop=True)


def run_external(match_ids, xt_grid, params, horizon_s=10, step_s=1.0, nx=33, ny=21, data_dir=SC_DIR, min_detected=0):
    """Returns (labelled table, AUC summary). min_detected keeps only frames where at least that many
    of the players on the pitch were actually seen on screen (the rest are extrapolated)."""
    table = external_labelled_table(match_ids, xt_grid, params, horizon_s, step_s, nx, ny, data_dir, min_detected)
    return table, _summarise(table, block_frames=BLOCK_FRAMES)


def detection_bins(table, bins=((0, 8), (8, 12), (12, 16), (16, 20), (20, 23))):
    """AUC of the main scores in bands of how many players were actually seen on screen."""
    rows = []
    for lo, hi in bins:
        sub = table[(table["n_detected"] >= lo) & (table["n_detected"] < hi)]
        if int(sub["label"].sum()) < 10:
            continue
        cols = [("share_tuned", "look-ahead"), ("share", "published"), ("depth", "depth"), ("ball_xt", "xT at ball")]
        if "gnn" in table.columns:
            cols.append(("gnn", "graph network"))
        rows.append({"detected": f"{lo}-{hi - 1}", "frames": len(sub), "positive": int(sub["label"].sum()),
                     **{name: auc(sub[col], sub["label"]) for col, name in cols}})
    return pd.DataFrame(rows)


def paired_by_detection(table, thresholds=(12, 16), n_boot=500):
    """Paired AUC differences (look-ahead vs published, vs team depth) on well-tracked frames only."""
    rows = []
    for k in thresholds:
        sub = table[table["n_detected"] >= k]
        for a, b, name in (("share_tuned", "share", "look-ahead vs published"),
                           ("share_tuned", "depth", "look-ahead vs team depth")):
            d, lo, hi, _ = paired_bootstrap(sub, a, b, n_boot=n_boot, block_frames=BLOCK_FRAMES)
            rows.append(dict(min_detected=k, comparison=name, frames=len(sub), auc_diff=d, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def paired_external(table):
    return paired_summary(table, block_frames=BLOCK_FRAMES)