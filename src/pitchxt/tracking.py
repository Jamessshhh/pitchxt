"""Loader for Metrica Sports sample tracking data (games 1 and 2, CSV format).

Output coordinates are in metres with the origin at the centre spot:
x in [-53, 53] (along the pitch), y in [-34, 34] (across the pitch).
"""
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICA_DIR = ROOT / "data" / "raw" / "metrica" / "data"

# Metrica doesn't ship pitch dimensions; 106 x 68 m is the usual assumption.
PITCH_LENGTH = 106.0
PITCH_WIDTH = 68.0

_COORD_COL = re.compile(r"_v?[xy]$")  # matches _x, _y, _vx, _vy


def _read_team(game_id, team, data_dir):
    path = (Path(data_dir) / f"Sample_Game_{game_id}"
            / f"Sample_Game_{game_id}_RawTrackingData_{team}_Team.csv")
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)                                   # row 1: team name
        jerseys = [x for x in next(reader) if x != ""]  # row 2: jersey numbers
        columns = next(reader)                          # row 3: column headers
    prefix = team.lower()
    for i, j in enumerate(jerseys):
        columns[3 + 2 * i] = f"{prefix}_{j}_x"
        columns[4 + 2 * i] = f"{prefix}_{j}_y"
    columns[-2] = "ball_x"
    columns[-1] = "ball_y"
    return pd.read_csv(path, names=columns, skiprows=3, index_col="Frame")


def load_tracking(game_id, data_dir=METRICA_DIR):
    """Home + away tracking merged on frame, converted to metres (centre origin)."""
    home = _read_team(game_id, "Home", data_dir)
    away = _read_team(game_id, "Away", data_dir)
    away = away.drop(columns=["Period", "Time [s]", "ball_x", "ball_y"])
    df = home.join(away).rename(columns={"Period": "period", "Time [s]": "time"})

    xcols = [c for c in df.columns if c.endswith("_x")]
    ycols = [c for c in df.columns if c.endswith("_y")]
    df[xcols] = (df[xcols] - 0.5) * PITCH_LENGTH
    df[ycols] = -(df[ycols] - 0.5) * PITCH_WIDTH   # Metrica y points down
    return df


def player_ids(df):
    """e.g. ['home_1', 'home_2', ..., 'away_25', ...]"""
    return sorted({c[:-2] for c in df.columns
                   if c.startswith(("home_", "away_")) and c.endswith("_x")})


def add_velocities(df, smooth=5, max_speed=12.0):
    """Add {player}_vx / {player}_vy columns in m/s.

    Centred finite difference per half, smoothed with a short rolling mean.
    Speeds above max_speed (m/s) are tracking glitches and are set to NaN.
    """
    period = df["period"]
    gt = df["time"].groupby(period)
    dt = gt.shift(-1) - gt.shift(1)

    new = {}
    for p in player_ids(df):
        for ax in ("x", "y"):
            pos = df[f"{p}_{ax}"].groupby(period)
            v = (pos.shift(-1) - pos.shift(1)) / dt
            v = v.groupby(period).transform(
                lambda s: s.rolling(smooth, center=True, min_periods=1).mean())
            new[f"{p}_v{ax}"] = v
        speed = np.hypot(new[f"{p}_vx"], new[f"{p}_vy"])
        bad = speed > max_speed
        new[f"{p}_vx"] = new[f"{p}_vx"].mask(bad)
        new[f"{p}_vy"] = new[f"{p}_vy"].mask(bad)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def to_single_direction(df):
    """Flip the second half so the home team always attacks toward +x."""
    out = df.copy()
    cols = [c for c in out.columns if _COORD_COL.search(c)]
    second = out["period"] == 2
    out.loc[second, cols] = -out.loc[second, cols]
    return out


def frame_players(df, frame, team):
    """Positions and velocities of one team at one frame, as an (n, 4) array
    with columns x, y, vx, vy. Players off the pitch / without data are dropped."""
    row = df.loc[frame]
    rows = []
    for p in player_ids(df):
        if not p.startswith(team):
            continue
        vals = [row[f"{p}_x"], row[f"{p}_y"], row[f"{p}_vx"], row[f"{p}_vy"]]
        if not np.isnan(vals[:2]).any():
            rows.append(vals)
    arr = np.array(rows, dtype=float)
    return np.nan_to_num(arr, nan=0.0) if len(arr) else arr.reshape(0, 4)