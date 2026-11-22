"""StatsBomb open-data loaders and event -> action reduction."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"

# Restarts are excluded by default because they're a different game state from open play.
SET_PIECE_PATTERNS = {
    "From Corner", "From Free Kick", "From Throw In", "From Goal Kick", "From Kick Off",
}


def load_events(competition_id, season_id, max_matches=None):
    """Download raw StatsBomb events for a competition-season."""
    from statsbombpy import sb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsbombpy warns about missing credentials
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        ids = matches["match_id"].tolist()[:max_matches]
        frames = []
        for i, mid in enumerate(ids, 1):
            print(f"[{i}/{len(ids)}] match {mid}")
            frames.append(sb.events(match_id=mid))
    return pd.concat(frames, ignore_index=True)


def _split_xy(col):
    """Turn a column of [x, y] lists (NaN where missing) into two float arrays."""
    arr = np.full((len(col), 2), np.nan)
    mask = col.apply(lambda v: isinstance(v, (list, tuple)) and len(v) >= 2).to_numpy()
    if mask.any():
        arr[mask] = np.array([v[:2] for v in col[mask]], dtype=float)
    return arr[:, 0], arr[:, 1]


def to_actions(events, open_play_only=True):
    """Reduce raw events to passes, carries and shots with start/end coordinates.

    StatsBomb coordinates are normalised so the team in possession always
    attacks toward x = 120.
    """
    ev = events[events["type"].isin(["Pass", "Carry", "Shot"])].copy()

    if open_play_only and "play_pattern" in ev:
        ev = ev[~ev["play_pattern"].isin(SET_PIECE_PATTERNS)]
    if "shot_type" in ev:
        ev = ev[~((ev["type"] == "Shot") & (ev["shot_type"] == "Penalty"))]

    is_pass = (ev["type"] == "Pass").to_numpy()
    is_carry = (ev["type"] == "Carry").to_numpy()
    is_shot = (ev["type"] == "Shot").to_numpy()

    x0, y0 = _split_xy(ev["location"])

    end_col = ev["pass_end_location"].where(ev["type"] == "Pass", ev["carry_end_location"])
    x1, y1 = _split_xy(end_col)
    x1 = np.where(is_shot, x0, x1)
    y1 = np.where(is_shot, y0, y1)

    success = np.where(is_pass, ev["pass_outcome"].isna().to_numpy(), is_carry)
    goal = is_shot & (ev["shot_outcome"] == "Goal").to_numpy()

    out = pd.DataFrame({
        "kind": np.where(is_shot, "shot", "move"),
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "success": success.astype(bool),
        "goal": goal,
    }, index=ev.index)

    for c in ["match_id", "team", "player", "period", "minute", "second"]:
        if c in ev:
            out[c] = ev[c]

    return out.dropna(subset=["x0", "y0", "x1", "y1"]).reset_index(drop=True)


def load_actions(competition_id, season_id, max_matches=None, refresh=False):
    """Load actions for a competition-season, cached to data/processed as parquet."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    tag = max_matches or "all"
    path = PROC_DIR / f"actions_{competition_id}_{season_id}_{tag}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    actions = to_actions(load_events(competition_id, season_id, max_matches))
    actions.to_parquet(path, index=False)
    return actions