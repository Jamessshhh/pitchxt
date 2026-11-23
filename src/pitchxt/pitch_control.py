"""Spearman (2018) pitch control model, vectorised with numpy.

For every target location on the pitch we estimate the probability that the
attacking team would control the ball if it arrived there, given where every
player is and how they are moving.
"""
import matplotlib.pyplot as plt
import numpy as np

from .pitch import draw_pitch_metres
from .tracking import PITCH_LENGTH, PITCH_WIDTH, frame_players

DEFAULT_PARAMS = dict(
    max_player_speed=5.0,      # m/s
    reaction_time=0.7,         # s before a player starts moving to the target
    tti_sigma=0.45,            # s, uncertainty in time-to-intercept
    lambda_att=4.3,            # control rate (1/s) once an attacker can reach the ball
    kappa_def=1.0,             # defenders' rate = lambda_att * kappa_def
    average_ball_speed=15.0,   # m/s
    dt=0.04,                   # s, integration step
    max_int_time=10.0,         # s
    tol=0.01,                  # stop once this little probability is unassigned
)


def make_grid(nx=50, ny=32, length=PITCH_LENGTH, width=PITCH_WIDTH):
    """Cell-centre coordinates of an nx x ny grid covering the pitch."""
    dx, dy = length / nx, width / ny
    xs = -length / 2 + dx * (np.arange(nx) + 0.5)
    ys = -width / 2 + dy * (np.arange(ny) + 0.5)
    return xs, ys


def _time_to_intercept(players, tx, ty, p):
    """(n_players, n_targets) time for each player to reach each target."""
    if len(players) == 0:
        return np.empty((0, tx.size))
    rx = players[:, 0] + players[:, 2] * p["reaction_time"]   # where they'll be after reacting
    ry = players[:, 1] + players[:, 3] * p["reaction_time"]
    dist = np.hypot(tx[None, :] - rx[:, None], ty[None, :] - ry[:, None])
    return p["reaction_time"] + dist / p["max_player_speed"]


def pitch_control_xy(att, dfn, ball_xy, xs, ys, params=None):
    """Pitch control for arrays of attackers and defenders.

    att, dfn : (n, 4) arrays with columns x, y, vx, vy
    ball_xy  : (x, y) of the ball
    Returns (pc_att, pc_def), each of shape (len(ys), len(xs)).
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    gx, gy = np.meshgrid(xs, ys)
    tx, ty = gx.ravel(), gy.ravel()

    ball_t = np.hypot(tx - ball_xy[0], ty - ball_xy[1]) / p["average_ball_speed"]
    tti_a = _time_to_intercept(att, tx, ty, p)
    tti_d = _time_to_intercept(dfn, tx, ty, p)

    lam_a = p["lambda_att"]
    lam_d = p["lambda_att"] * p["kappa_def"]
    k_sig = np.pi / (np.sqrt(3.0) * p["tti_sigma"])

    pc_a = np.zeros(tx.size)
    pc_d = np.zeros(tx.size)

    for k in range(1, int(p["max_int_time"] / p["dt"]) + 1):
        T = ball_t + k * p["dt"]
        # probability each player can have reached the ball by time T
        f_a = (1.0 / (1.0 + np.exp(np.clip(-k_sig * (T[None, :] - tti_a), -50, 50)))).sum(axis=0)
        f_d = (1.0 / (1.0 + np.exp(np.clip(-k_sig * (T[None, :] - tti_d), -50, 50)))).sum(axis=0)

        rate_a, rate_d = lam_a * f_a, lam_d * f_d
        rate = rate_a + rate_d
        remaining = 1.0 - pc_a - pc_d

        # exact update of dP/dT = remaining * rate over one step (stable even with many players)
        newly = remaining * (1.0 - np.exp(-rate * p["dt"]))
        with np.errstate(divide="ignore", invalid="ignore"):
            share_a = np.where(rate > 0, rate_a / rate, 0.0)
        pc_a += newly * share_a
        pc_d += newly * (1.0 - share_a)

        if (1.0 - pc_a - pc_d).max() < p["tol"]:
            break

    shape = (len(ys), len(xs))
    return pc_a.reshape(shape), pc_d.reshape(shape)


def pitch_control_frame(df, frame, attacking_team="home", nx=50, ny=32, params=None):
    """Pitch control for one tracking frame. Returns xs, ys, pc_att, pc_def."""
    defending_team = "away" if attacking_team == "home" else "home"
    att = frame_players(df, frame, attacking_team)
    dfn = frame_players(df, frame, defending_team)
    row = df.loc[frame]
    ball = (row["ball_x"], row["ball_y"])
    if np.isnan(ball).any():
        raise ValueError(f"No ball position at frame {frame}")
    xs, ys = make_grid(nx, ny)
    pc_a, pc_d = pitch_control_xy(att, dfn, ball, xs, ys, params)
    return xs, ys, pc_a, pc_d


def plot_pitch_control(df, frame, attacking_team="home", nx=50, ny=32, ax=None, params=None):
    """Draw the pitch control surface with players, velocities and the ball."""
    xs, ys, pc_a, _ = pitch_control_frame(df, frame, attacking_team, nx, ny, params)
    ax = draw_pitch_metres(ax)
    ax.imshow(pc_a, origin="lower", cmap="bwr", vmin=0, vmax=1, alpha=0.6, zorder=0,
              extent=(-PITCH_LENGTH / 2, PITCH_LENGTH / 2, -PITCH_WIDTH / 2, PITCH_WIDTH / 2))

    defending_team = "away" if attacking_team == "home" else "home"
    for team, colour in ((attacking_team, "#b00020"), (defending_team, "#1a3fb0")):
        pl = frame_players(df, frame, team)
        ax.scatter(pl[:, 0], pl[:, 1], s=140, c=colour, ec="white", lw=1.5, zorder=3)
        ax.quiver(pl[:, 0], pl[:, 1], pl[:, 2], pl[:, 3], color=colour,
                  angles="xy", scale_units="xy", scale=1, width=0.003, zorder=2)
    ax.scatter(df.loc[frame, "ball_x"], df.loc[frame, "ball_y"], s=60, c="black", ec="white", zorder=4)
    ax.set_title(f"Pitch control, frame {frame} (red = {attacking_team})")
    return ax