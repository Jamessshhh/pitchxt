"""Turn tracking frames into fixed-size arrays for the graph model. No PyTorch needed here.

One record per sampled frame (the same frames as the hand-built scores: one per second while the ball
is in play and tracked): up to 12 players per team (x, y, vx, vy, padded with a mask), the ball
(x, y, vx, vy), and 'shot within N seconds' labels for each team. Positions follow the pitchxt
convention (metres, centre-origin, home attacking toward +x).
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .matches import load_match, load_shots
from .tracking import player_ids

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
MAX_PER_TEAM = 12
HORIZONS = (5, 10, 15)
_KEYS = ("x", "y", "vx", "vy")


def sample_positions(df, step):
    """Row positions of the frames to use: every `step` rows, ball in play, ball position known."""
    alive = df["ball_alive"].to_numpy()
    ball_ok = df["ball_x"].notna().to_numpy() & df["ball_y"].notna().to_numpy()
    pos = np.arange(0, len(df), step)
    return pos[alive[pos] & ball_ok[pos]]


def _team_arrays(df, team, pos, max_players=MAX_PER_TEAM):
    """(n, max_players, 4) array of x, y, vx, vy and a (n, max_players) mask of real players."""
    ids = [p for p in player_ids(df) if p.startswith(team)]
    cols = {k: df[[f"{p}_{k}" for p in ids]].to_numpy()[pos] for k in _KEYS}
    valid = ~(np.isnan(cols["x"]) | np.isnan(cols["y"]))
    order = np.argsort(~valid, axis=1, kind="stable")[:, :max_players]     # real players first
    n, k = len(pos), order.shape[1]
    out = np.zeros((n, max_players, 4), dtype=np.float32)
    mask = np.zeros((n, max_players), dtype=bool)
    mask[:, :k] = np.take_along_axis(valid, order, axis=1)
    for j, key in enumerate(_KEYS):
        vals = np.take_along_axis(np.nan_to_num(cols[key]), order, axis=1)
        out[:, :k, j] = np.where(mask[:, :k], vals, 0.0)
    return out, mask


def _ball_arrays(df, pos, fps):
    """(n, 4) ball x, y, vx, vy (velocity from a +-0.2 s difference, zero where unknown)."""
    bx, by = df["ball_x"].to_numpy(), df["ball_y"].to_numpy()
    k = max(1, int(round(0.2 * fps)))
    lo, hi = np.clip(pos - k, 0, len(df) - 1), np.clip(pos + k, 0, len(df) - 1)
    dt = np.maximum(hi - lo, 1) / fps
    vx, vy = np.nan_to_num((bx[hi] - bx[lo]) / dt), np.nan_to_num((by[hi] - by[lo]) / dt)
    return np.stack([bx[pos], by[pos], vx, vy], axis=1).astype(np.float32)


def shot_labels(frames, shot_frames, horizon_frames):
    """True where a shot starts within (frame, frame + horizon_frames]."""
    s = np.asarray(shot_frames)[None, :]
    f = np.asarray(frames)[:, None]
    return ((s > f) & (s <= f + horizon_frames)).any(axis=1)


def build_from_tracking(df, shots, fps, step_s=1.0):
    """df: a tracking table prepared like the rest of pitchxt (velocities added, direction normalised)."""
    step = max(1, int(round(fps * step_s)))
    pos = sample_positions(df, step)
    frames = df.index.to_numpy()[pos].astype(np.int64)
    home, hm = _team_arrays(df, "home", pos)
    away, am = _team_arrays(df, "away", pos)
    out = dict(frames=frames, players=np.stack([home, away], axis=1), mask=np.stack([hm, am], axis=1),
               ball=_ball_arrays(df, pos, fps), period=df["period"].to_numpy()[pos].astype(np.int8),
               n_detected=(df["n_detected"].to_numpy()[pos] if "n_detected" in df
                           else np.full(len(pos), 22)).astype(np.int16))
    for team in ("home", "away"):
        s = shots.loc[shots["team"] == team, "start_frame"].to_numpy()
        for h in HORIZONS:
            out[f"y_{team}_{h}"] = shot_labels(frames, s, int(h * fps))
    return out


def _cache_path(name, step_s):
    return PROC_DIR / f"gnn_{name}_step{step_s:g}s.npz"


def _load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def build_graph_data(game_id, step_s=1.0, refresh=False):
    """Metrica / DFL match -> arrays (cached). 25 fps."""
    path = _cache_path(game_id, step_s)
    if path.exists() and not refresh:
        return _load_npz(path)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df, _ = load_match(game_id)
    data = build_from_tracking(df, load_shots(game_id), fps=25, step_s=step_s)
    np.savez_compressed(path, **data)
    return data


def build_graph_data_sc(match_id, step_s=1.0, data_dir=None, refresh=False):
    """SkillCorner match -> arrays (cached). 10 fps; also carries n_detected."""
    from .skillcorner import FPS, SC_DIR, load_skillcorner_match
    from .tracking import add_velocities, to_single_direction

    path = _cache_path(f"sc{match_id}", step_s)
    if path.exists() and not refresh:
        return _load_npz(path)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    raw, shots = load_skillcorner_match(match_id, data_dir or SC_DIR)
    data = build_from_tracking(to_single_direction(add_velocities(raw)), shots, fps=FPS, step_s=step_s)
    np.savez_compressed(path, **data)
    return data


def _worker(args):
    kind, game_id, step_s = args
    (build_graph_data_sc if kind == "sc" else build_graph_data)(game_id, step_s)
    return str(game_id)


def precompute_graph_data(game_ids, step_s=1.0, sc_ids=(), workers=3):
    """Fill the caches in parallel (call from under `if __name__ == "__main__"`)."""
    jobs = [("main", g, step_s) for g in game_ids if not _cache_path(g, step_s).exists()]
    jobs += [("sc", m, step_s) for m in sc_ids if not _cache_path(f"sc{m}", step_s).exists()]
    if not jobs:
        return
    print(f"building graph data for {len(jobs)} matches with {workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for done in ex.map(_worker, jobs):
            print(f"  finished {done}", flush=True)