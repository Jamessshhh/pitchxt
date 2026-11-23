"""Combine pitch control with expected threat: the value of the space each team controls."""
import matplotlib.pyplot as plt
import numpy as np

from .pitch import draw_pitch_metres
from .pitch_control import pitch_control_frame
from .tracking import PITCH_LENGTH, PITCH_WIDTH, frame_players
from .xt import xt_lookup

SB_L, SB_W = 120.0, 80.0  # StatsBomb pitch size the xT grid was fitted on


def xt_on_control_grid(xt_grid, xs, ys, attacking_team="home"):
    """xT value of every pitch-control cell, from the point of view of `attacking_team`.

    Tracking coordinates are centre-origin metres with the home team attacking
    toward +x. xT is defined for a team attacking toward x = 120, so the away
    team's view is a 180 degree rotation.
    """
    gx, gy = np.meshgrid(xs, ys)
    if attacking_team == "away":
        gx, gy = -gx, -gy
    x_sb = (gx + PITCH_LENGTH / 2) / PITCH_LENGTH * SB_L
    y_sb = (PITCH_WIDTH / 2 - gy) / PITCH_WIDTH * SB_W   # StatsBomb y runs top to bottom
    return xt_lookup(xt_grid, x_sb, y_sb)


def frame_value(df, frame, xt_grid, nx=50, ny=32, params=None):
    """Value of controlled space for both teams at one frame.

    Returns a dict with the grids and these scalars:
      value_home / value_away : mean over the pitch of control x xT, each team
                                measured with its own attacking direction
      threat_share            : value_home / (value_home + value_away), 0.5 = balanced
    """
    xs, ys, pc_h, pc_a = pitch_control_frame(df, frame, "home", nx, ny, params)
    xt_h = xt_on_control_grid(xt_grid, xs, ys, "home")
    xt_a = xt_on_control_grid(xt_grid, xs, ys, "away")
    val_h, val_a = pc_h * xt_h, pc_a * xt_a
    vh, va = val_h.mean(), val_a.mean()
    return dict(
        xs=xs, ys=ys, pc_home=pc_h, pc_away=pc_a, value_map_home=val_h, value_map_away=val_a,
        value_home=vh, value_away=va,
        threat_share=vh / (vh + va) if (vh + va) > 0 else np.nan,
    )


def value_series(df, xt_grid, frames, nx=33, ny=21, params=None):
    """threat_share (and both values) over a list of frames. Returns a DataFrame."""
    import pandas as pd

    rows = []
    for f in frames:
        try:
            r = frame_value(df, f, xt_grid, nx, ny, params)
        except ValueError:      # no ball position at this frame
            continue
        rows.append((f, df.loc[f, "period"], df.loc[f, "time"],
                     r["value_home"], r["value_away"], r["threat_share"]))
    return pd.DataFrame(rows, columns=["frame", "period", "time", "value_home", "value_away", "threat_share"])


def plot_value_map(df, frame, xt_grid, team="home", nx=50, ny=32, ax=None, params=None):
    """Heatmap of (control x xT) for one team, with players and the ball."""
    r = frame_value(df, frame, xt_grid, nx, ny, params)
    ax = draw_pitch_metres(ax)
    im = ax.imshow(r[f"value_map_{team}"], origin="lower", cmap="magma_r", alpha=0.75, zorder=0,
                   extent=(-PITCH_LENGTH / 2, PITCH_LENGTH / 2, -PITCH_WIDTH / 2, PITCH_WIDTH / 2))
    other = "away" if team == "home" else "home"
    for t, colour in ((team, "#b00020"), (other, "#1a3fb0")):
        pl = frame_players(df, frame, t)
        ax.scatter(pl[:, 0], pl[:, 1], s=120, c=colour, ec="white", lw=1.5, zorder=3)
    ax.scatter(df.loc[frame, "ball_x"], df.loc[frame, "ball_y"], s=60, c="black", ec="white", zorder=4)
    ax.set_title(f"Value of controlled space, {team} (frame {frame}), threat share {r['threat_share']:.2f}")
    plt.colorbar(im, ax=ax, shrink=0.6, label="control x xT")
    return ax