"""One entry point for loading a match from any supported source (Metrica games 1-2, DFL matches)."""
from pathlib import Path

import pandas as pd

from .dfl import MATCH_IDS as DFL_IDS
from .dfl import load_dfl_match
from .events import get_shots, load_metrica_events
from .tracking import add_velocities, load_tracking, to_single_direction

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"


def source_of(game_id):
    """'dfl' for DFL match ids such as 'J03WPY', otherwise 'metrica'."""
    return "dfl" if str(game_id) in DFL_IDS else "metrica"


def _shots_path(game_id):
    return PROC_DIR / f"shots_{game_id}.parquet"


def load_match(game_id):
    """Returns (tracking, shots). Tracking is ready for pitch control: metres, velocities added,
    home attacking toward +x, with a `ball_alive` column (always True for Metrica, which has no
    dead-ball flag). Shots have team, period, start_frame (an index value of tracking) and goal."""
    if source_of(game_id) == "metrica":
        tracking = to_single_direction(add_velocities(load_tracking(int(game_id))))
        tracking["ball_alive"] = True
        return tracking, get_shots(load_metrica_events(int(game_id)))

    raw, shots = load_dfl_match(str(game_id))
    tracking = to_single_direction(add_velocities(raw))
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    shots.to_parquet(_shots_path(game_id), index=False)       # so shots can be reloaded without re-parsing
    return tracking, shots


def load_shots(game_id):
    """Shots only. For DFL matches this reads the cache written by load_match when it exists."""
    if source_of(game_id) == "metrica":
        return get_shots(load_metrica_events(int(game_id)))
    path = _shots_path(game_id)
    return pd.read_parquet(path) if path.exists() else load_match(game_id)[1]