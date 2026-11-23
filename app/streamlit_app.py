"""Interactive explorer: streamlit run app/streamlit_app.py"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from pitchxt import fit_xt
from pitchxt.data import load_actions
from pitchxt.events import get_shots, load_metrica_events
from pitchxt.pitch_control import plot_pitch_control
from pitchxt.tracking import add_velocities, load_tracking, to_single_direction
from pitchxt.validate import FPS, frame_table
from pitchxt.value import frame_value, plot_value_map

st.set_page_config(page_title="pitchxt", layout="wide")


@st.cache_resource(show_spinner="Fitting xT (the first run downloads World Cup 2022 data, a few minutes)...")
def get_xt_grid():
    return fit_xt(load_actions(43, 106))


@st.cache_resource(show_spinner="Loading tracking data...")
def get_game(game_id):
    df = to_single_direction(add_velocities(load_tracking(game_id)))
    shots = get_shots(load_metrica_events(game_id))
    return df, shots


@st.cache_data(show_spinner="Computing threat share through the match (first time only, about a minute)...")
def get_timeline(game_id, _grid):
    t = frame_table(game_id, _grid)
    return t[t.team == "home"].sort_values("frame")[["frame", "share"]]


st.title("pitchxt: who controls the dangerous space?")
st.caption("Pitch control (Spearman) × expected threat on Metrica sample tracking data. "
           "Home attacks to the right in every frame.")

grid = get_xt_grid()
game = st.sidebar.selectbox("Game", [1, 2])
df, shots = get_game(game)
last = int(df.index.max())

mode = st.sidebar.radio("Pick a moment", ["Browse by time", "Jump to a shot"])
if mode == "Browse by time":
    t = st.sidebar.slider("Time (s)", 0, int(df["time"].max()), 600, step=1)
    frame = int(np.clip(round(t * FPS) + 1, 1, last))
else:
    labels = [f"{r.team} shot, minute {r.start_time / 60:.1f}, {r.subtype}" for r in shots.itertuples()]
    i = st.sidebar.selectbox("Shot", range(len(shots)), format_func=lambda k: labels[k])
    lag = st.sidebar.slider("Seconds before the shot", 0, 15, 5)
    frame = int(np.clip(shots.start_frame[i] - lag * FPS, 1, last))

view = st.sidebar.radio("Map", ["Pitch control", "Value of space (control × xT)"])
team = st.sidebar.radio("Shown from the point of view of", ["home", "away"])

try:
    r = frame_value(df, frame, grid)
except ValueError:
    st.warning("The ball isn't tracked in this frame. Pick another moment.")
    st.stop()

left, right = st.columns([3, 1])
with left:
    fig, ax = plt.subplots(figsize=(9, 6))
    if view == "Pitch control":
        plot_pitch_control(df, frame, attacking_team=team, ax=ax)
    else:
        plot_value_map(df, frame, grid, team, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.metric("Home threat share", f"{r['threat_share']:.2f}",
              help="Home share of the dangerous controlled space. 0.5 means balanced.")
    st.metric("Home value of space", f"{r['value_home']:.4f}")
    st.metric("Away value of space", f"{r['value_away']:.4f}")
    soon = shots[(shots.start_frame > frame) & (shots.start_frame <= frame + 10 * FPS)]
    st.write(f"**Shots in the next 10 s:** home {int((soon.team == 'home').sum())}, "
             f"away {int((soon.team == 'away').sum())}")
    st.write(f"Frame {frame} · {frame / FPS:.1f} s")

tl = get_timeline(game, grid)
fig, ax = plt.subplots(figsize=(14, 3.2))
minutes = tl["frame"] / FPS / 60
ax.plot(minutes, tl["share"].rolling(30, center=True, min_periods=1).mean(), color="#333", lw=1.4)
ax.axhline(0.5, color="#999", lw=1, ls="--")
for tm, colour, y, mk in (("home", "#b00020", 0.96, "v"), ("away", "#1a3fb0", 0.04, "^")):
    s = shots[shots.team == tm]
    ax.scatter(s.start_frame / FPS / 60, [y] * len(s), c=colour, s=40, marker=mk, label=f"{tm} shot", zorder=3)
ax.axvline(frame / FPS / 60, color="#e08a00", lw=2)
ax.set_ylim(0, 1)
ax.set_xlabel("minute of the recording")
ax.set_ylabel("home threat share")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
st.pyplot(fig)
plt.close(fig)