"""Loader for Metrica Sports sample event data (games 1 and 2, CSV format)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICA_DIR = ROOT / "data" / "raw" / "metrica" / "data"

_RENAME = {
    "Team": "team", "Type": "type", "Subtype": "subtype", "Period": "period",
    "Start Frame": "start_frame", "Start Time [s]": "start_time",
    "End Frame": "end_frame", "End Time [s]": "end_time",
    "From": "from_player", "To": "to_player",
}


def load_metrica_events(game_id, data_dir=METRICA_DIR):
    """Events for one game: team is 'home'/'away', frames match the tracking frame index."""
    path = Path(data_dir) / f"Sample_Game_{game_id}" / f"Sample_Game_{game_id}_RawEventsData.csv"
    ev = pd.read_csv(path).rename(columns=_RENAME)
    ev["team"] = ev["team"].str.lower()
    return ev[list(_RENAME.values())]


def get_shots(events):
    """All shots, with a boolean 'goal' column."""
    s = events[events["type"] == "SHOT"].copy()
    s["goal"] = s["subtype"].fillna("").str.contains("GOAL")
    return s.reset_index(drop=True)