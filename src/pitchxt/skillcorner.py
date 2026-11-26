"""Loader for the SkillCorner open data (A-League 2024/25, broadcast tracking at 10 fps).

Same output layout as the other loaders (columns home_<shirt>_x/_y, away_<shirt>_x/_y, ball_x/_y,
period, time, ball_alive; centre-origin metres), plus `n_detected`: how many players were actually
seen on screen in that frame (the others are extrapolated by SkillCorner). The index is the
SkillCorner frame number. Shots come from the dynamic events: a possession that ends in a shot.

Get the data with Git LFS (the tracking files are LFS objects, a plain `git clone` gives pointers):
    brew install git-lfs && git lfs install
    git clone --depth 1 https://github.com/SkillCorner/opendata.git data/raw/skillcorner
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SC_DIR = ROOT / "data" / "raw" / "skillcorner" / "data"
FPS = 10
MIN_TRACKING_BYTES = 1_000_000      # Git LFS pointer files are ~130 bytes; real tracking files are ~90 MB


def _paths(match_id, data_dir):
    d = Path(data_dir) / "matches" / str(match_id)
    return (d / f"{match_id}_match.json", d / f"{match_id}_tracking_extrapolated.jsonl",
            d / f"{match_id}_dynamic_events.csv", d / f"{match_id}_phases_of_play.csv")


def downloaded_skillcorner(data_dir=SC_DIR):
    """Match ids whose tracking file is real data, not a Git LFS pointer."""
    out = []
    for d in sorted(Path(data_dir, "matches").glob("*")):
        meta, trk, evt, _ = _paths(d.name, data_dir)
        if meta.exists() and evt.exists() and trk.exists() and trk.stat().st_size > MIN_TRACKING_BYTES:
            out.append(d.name)
    return out


def _f(v):
    return float(v) if v is not None else np.nan


def _id_maps(match):
    """player id -> 'home_<shirt>' / 'away_<shirt>', keyed two ways (tracking may use either id)."""
    home_id = match["home_team"]["id"]
    by_track, by_id = {}, {}
    for p in match["players"]:
        prefix = f"{'home' if p['team_id'] == home_id else 'away'}_{p['number']}"
        by_track[p["trackable_object"]] = prefix
        by_id[p["id"]] = prefix
    return by_track, by_id


def _parse_tracking(rows, mapping, length, width, phases):
    """rows: parsed jsonl frames (with a period). Returns the pitchxt tracking table."""
    rows = sorted(rows, key=lambda r: r["frame"])
    n = len(rows)
    prefixes = sorted(set(mapping.values()))
    col = {pid: prefixes.index(pre) for pid, pre in mapping.items()}
    X = np.full((n, len(prefixes)), np.nan)
    Y = np.full((n, len(prefixes)), np.nan)
    D = np.zeros((n, len(prefixes)), dtype=bool)
    bx, by = np.full(n, np.nan), np.full(n, np.nan)
    for i, r in enumerate(rows):
        ball = r.get("ball_data") or {}
        bx[i], by[i] = _f(ball.get("x")), _f(ball.get("y"))
        for p in r.get("player_data") or []:
            j = col.get(p.get("player_id"))
            if j is None:
                continue
            X[i, j], Y[i, j], D[i, j] = _f(p.get("x")), _f(p.get("y")), bool(p.get("is_detected"))
    frames = np.array([r["frame"] for r in rows], dtype=np.int64)

    alive = np.ones(n, dtype=bool)
    if phases is not None:                       # phases of play exist only while the ball is in play
        alive = np.zeros(n, dtype=bool)
        for s, e in zip(phases["frame_start"], phases["frame_end"]):
            alive[np.searchsorted(frames, s, "left"):np.searchsorted(frames, e, "right")] = True

    cols = {"period": np.array([r["period"] for r in rows]), "time": frames / FPS,
            "ball_x": bx, "ball_y": by, "ball_alive": alive, "n_detected": D.sum(axis=1)}
    for j, pre in enumerate(prefixes):
        cols[f"{pre}_x"], cols[f"{pre}_y"] = X[:, j], Y[:, j]
    out = pd.DataFrame(cols, index=pd.Index(frames, name="Frame"))
    return out[~out.index.duplicated()]


def _load_shots(evt_path, home_id):
    ev = pd.read_csv(evt_path, low_memory=False,
                     usecols=["event_type", "end_type", "team_id", "frame_end", "period", "lead_to_goal", "x_end", "y_end"])
    pp = ev[(ev["event_type"] == "player_possession") & (ev["end_type"] == "shot")]
    return pd.DataFrame({
        "team": np.where(pp["team_id"] == home_id, "home", "away"),
        "period": pp["period"].astype(int).to_numpy(),
        "start_frame": pp["frame_end"].astype(int).to_numpy(),      # the shot is the end of the possession
        "goal": pp["lead_to_goal"].fillna(False).astype(bool).to_numpy(),
        "ev_x": pp["x_end"].to_numpy(), "ev_y": pp["y_end"].to_numpy(),
    }).reset_index(drop=True)


def load_skillcorner_match(match_id, data_dir=SC_DIR):
    """Returns (tracking, shots) in raw pitch orientation; run add_velocities and to_single_direction on it."""
    meta, trk, evt, phases_path = _paths(match_id, data_dir)
    if trk.stat().st_size <= MIN_TRACKING_BYTES:
        raise RuntimeError(f"{trk.name} looks like a Git LFS pointer, not data. Install git-lfs and run `git lfs pull`.")
    match = json.loads(meta.read_text())
    by_track, by_id = _id_maps(match)

    rows = []
    with open(trk) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("period") is not None:
                rows.append(r)
    seen = {p.get("player_id") for r in rows[:300] for p in (r.get("player_data") or [])}
    n_track, n_id = len(seen & set(by_track)), len(seen & set(by_id))
    mapping, key = (by_track, "trackable_object") if n_track >= n_id else (by_id, "id")
    if max(n_track, n_id) == 0:
        raise RuntimeError("None of the tracking player ids match the match file; the format may have changed.")

    phases = pd.read_csv(phases_path, usecols=["frame_start", "frame_end"]) if phases_path.exists() else None
    tracking = _parse_tracking(rows, mapping, float(match["pitch_length"]), float(match["pitch_width"]), phases)
    tracking.attrs.update(id_key=key, home_team_side=match.get("home_team_side"),
                          pitch=(match["pitch_length"], match["pitch_width"]))
    return tracking, _load_shots(evt, match["home_team"]["id"])


def load_skillcorner_shots(match_id, data_dir=SC_DIR):
    """Shots only: reads the small match and event files, not the tracking."""
    meta, _, evt, _ = _paths(match_id, data_dir)
    return _load_shots(evt, json.loads(meta.read_text())["home_team"]["id"])