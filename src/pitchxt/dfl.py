"""Loader for the DFL / Sportec open dataset (7 Bundesliga matches) via kloppy.

Produces the same table layout as tracking.load_tracking (columns home_<jersey>_x/_y,
away_<jersey>_x/_y, ball_x/_y, period, time; centre-origin metres, y up) so the rest
of pitchxt works unchanged. The index is the DFL frame id, which has a gap between
halves, and `time` is seconds since the start of the period.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DFL_DIR = ROOT / "data" / "raw" / "dfl"
FPS = 25
MATCH_IDS = ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]


def _find(data_dir, pattern, match_id):
    files = sorted(Path(data_dir).glob(pattern.format(match_id=match_id)))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern.format(match_id=match_id)} in {data_dir}")
    return files[0]


def downloaded_matches(data_dir=DFL_DIR):
    """Match ids whose three files (info, events, tracking) are all present."""
    out = []
    for m in MATCH_IDS:
        try:
            for pat in ("DFL_02_01_matchinformation_*_DFL-MAT-{match_id}.xml",
                        "DFL_03_02_events_raw_*_DFL-MAT-{match_id}.xml",
                        "DFL_04_03_positions_raw_observed_*_DFL-MAT-{match_id}.xml"):
                _find(data_dir, pat, m)
            out.append(m)
        except FileNotFoundError:
            pass
    return out


def _convert_tracking(raw, player_prefix, length, width):
    """kloppy dataframe (normalised, origin top-left, y down) -> pitchxt table (metres, centre, y up)."""
    cols = {
        "period": raw["period_id"].to_numpy(),
        "time": raw["timestamp"].dt.total_seconds().to_numpy(),
        "ball_x": (raw["ball_x"].to_numpy() - 0.5) * length,
        "ball_y": -(raw["ball_y"].to_numpy() - 0.5) * width,
        "ball_alive": (raw["ball_state"] == "alive").to_numpy(),
    }
    for pid, prefix in player_prefix.items():
        if f"{pid}_x" not in raw.columns:
            continue
        cols[f"{prefix}_x"] = (raw[f"{pid}_x"].to_numpy() - 0.5) * length
        cols[f"{prefix}_y"] = -(raw[f"{pid}_y"].to_numpy() - 0.5) * width
    return pd.DataFrame(cols, index=pd.Index(raw["frame_id"].to_numpy(), name="Frame"))


def _shots_to_frames(rows, tracking):
    """Attach `start_frame` (a tracking index value) to shot rows given as period + seconds."""
    shots = pd.DataFrame(rows, columns=["team", "period", "seconds", "result", "ev_x", "ev_y"])
    first = tracking.reset_index().groupby("period")["Frame"].min()
    shots["start_frame"] = shots["period"].map(first) + (shots["seconds"] * FPS).round().astype(int)
    shots["goal"] = shots["result"].str.contains("GOAL")
    return shots


def load_dfl_match(match_id, data_dir=DFL_DIR):
    """Returns (tracking, shots) for one match. Tracking is in raw pitch orientation:
    run add_velocities and to_single_direction on it as for Metrica."""
    from kloppy import sportec
    from kloppy.domain import Ground

    meta = _find(data_dir, "DFL_02_01_matchinformation_*_DFL-MAT-{match_id}.xml", match_id)
    trk = _find(data_dir, "DFL_04_03_positions_raw_observed_*_DFL-MAT-{match_id}.xml", match_id)
    evt = _find(data_dir, "DFL_03_02_events_raw_*_DFL-MAT-{match_id}.xml", match_id)

    ds = sportec.load_tracking(meta_data=str(meta), raw_data=str(trk))
    dims = ds.metadata.pitch_dimensions
    length, width = float(dims.pitch_length), float(dims.pitch_width)
    prefix = {p.player_id: f"{'home' if t.ground == Ground.HOME else 'away'}_{p.jersey_no}"
              for t in ds.metadata.teams for p in t.players}
    tracking = _convert_tracking(ds.to_df(), prefix, length, width)

    events = sportec.load_event(event_data=str(evt), meta_data=str(meta))
    rows = []
    for e in events.events:
        if e.event_type.name != "SHOT":
            continue
        c = e.coordinates
        rows.append(dict(
            team="home" if e.team.ground == Ground.HOME else "away",
            period=e.period.id,
            seconds=e.timestamp.total_seconds(),
            result=getattr(e.result, "name", str(e.result)),
            ev_x=(c.x - 0.5) * length if c is not None else np.nan,
            ev_y=-(c.y - 0.5) * width if c is not None else np.nan,
        ))
    return tracking, _shots_to_frames(rows, tracking)