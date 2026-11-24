"""Regenerate every figure used in the README: python scripts/make_figures.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pitchxt import fit_xt, plot_grid
from pitchxt.data import load_actions
from pitchxt.dfl import downloaded_matches
from pitchxt.events import get_shots, load_metrica_events
from pitchxt.matches import load_shots
from pitchxt.pitch_control import plot_pitch_control
from pitchxt.tracking import add_velocities, load_tracking, to_single_direction
from pitchxt.validate import run_validation
from pitchxt.value import plot_value_map

OUT = Path(__file__).resolve().parents[1] / "docs"
OUT.mkdir(exist_ok=True)
FRAME = 1000          # an illustrative frame from Metrica game 1
GAME = 1
GAMES = [1, 2] + downloaded_matches()   # Metrica games plus any DFL matches in data/raw/dfl


def save(fig, name):
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def main():
    grid = fit_xt(load_actions(43, 106))
    df = to_single_direction(add_velocities(load_tracking(GAME)))

    # 1. expected threat surface
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_grid(grid, ax=ax, title="Expected threat (xT), StatsBomb open data, World Cup 2022")
    save(fig, "xt_surface.png")

    # 2. pitch control
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_pitch_control(df, FRAME, ax=ax)
    save(fig, "pitch_control.png")

    # 3. value of controlled space, both teams
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    plot_value_map(df, FRAME, grid, "home", ax=axes[0])
    plot_value_map(df, FRAME, grid, "away", ax=axes[1])
    save(fig, "value_of_space.png")

    # 4. validation across horizons (+ keep the 10 s table for the timeline)
    horizons, results, table10 = (5, 10, 15), [], None
    for h in horizons:
        table, summary = run_validation(GAMES, grid, horizon_s=h)
        results.append(summary)
        if h == 10:
            table10 = table
    names = list(results[0]["score"])
    colours = ["#9aa5b1", "#5b8def", "#d6455d"]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.26
    for i, name in enumerate(names):
        aucs = np.array([r.loc[r.score == name, "auc"].iloc[0] for r in results])
        lo = np.array([r.loc[r.score == name, "ci_low"].iloc[0] for r in results])
        hi = np.array([r.loc[r.score == name, "ci_high"].iloc[0] for r in results])
        ax.bar(np.arange(len(horizons)) + (i - 1) * width, aucs, width, label=name, color=colours[i],
               yerr=[aucs - lo, hi - aucs], capsize=3, ecolor="#333")
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"shot within {h} s" for h in horizons])
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="#999", lw=1, ls="--")
    ax.set_ylabel("AUC (0.5 = chance)")
    n_shots = sum(len(load_shots(g)) for g in GAMES)
    ax.set_title(f"Predicting shots from a single frame: {len(GAMES)} matches, {n_shots} shots")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "validation_auc.png")

    # 5. threat share through game 1, with shots marked
    t = table10[(table10.game == str(GAME)) & (table10.team == "home")].sort_values("frame")
    minutes = t["frame"] / 25 / 60
    smooth = t["share"].rolling(30, center=True, min_periods=1).mean()
    shots = get_shots(load_metrica_events(GAME))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(minutes, smooth, color="#333", lw=1.5)
    ax.axhline(0.5, color="#999", lw=1, ls="--")
    for team, colour in (("home", "#b00020"), ("away", "#1a3fb0")):
        s = shots[shots.team == team]
        ax.scatter(s.start_frame / 25 / 60, [0.97 if team == "home" else 0.03] * len(s), c=colour, s=45,
                   marker="v" if team == "home" else "^", label=f"{team} shot", zorder=3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("minute of the recording")
    ax.set_ylabel("home threat share")
    ax.set_title("Who controls the dangerous space? Game 1 (30 s rolling mean)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "threat_share_game1.png")


if __name__ == "__main__":
    main()