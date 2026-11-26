"""Regenerate every figure used in the README: python scripts/make_figures.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pitchxt import fit_xt, plot_grid
from pitchxt.data import load_actions
from pitchxt.dfl import downloaded_matches
from pitchxt.events import get_shots, load_metrica_events
from pitchxt.external import detection_bins, external_labelled_table, run_external
from pitchxt.matches import load_shots
from pitchxt.pitch_control import plot_pitch_control
from pitchxt.plots import detection_chart, validation_chart
from pitchxt.skillcorner import downloaded_skillcorner, load_skillcorner_shots
from pitchxt.tracking import add_velocities, load_tracking, to_single_direction
from pitchxt.validate import TUNED_PARAMS, precompute_tuned, run_validation
from pitchxt.value import plot_value_map

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
OUT.mkdir(exist_ok=True)
FRAME = 1000          # an illustrative frame from Metrica game 1
GAME = 1
GAMES = [1, 2] + downloaded_matches()   # Metrica games plus any DFL matches in data/raw/dfl


def save(fig, name):
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def _external_summary(horizon_s, sc_ids, grid):
    """AUC summary on the SkillCorner matches, cached because the bootstrap is slow."""
    path = ROOT / "data" / "processed" / f"external_summary_{horizon_s}s.csv"
    if path.exists():
        return pd.read_csv(path)
    _, summary = run_external(sc_ids, grid, TUNED_PARAMS, horizon_s=horizon_s)
    summary.to_csv(path, index=False)
    return summary


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
    precompute_tuned(GAMES, grid, TUNED_PARAMS)
    for h in horizons:
        table, summary = run_validation(GAMES, grid, horizon_s=h, tuned=TUNED_PARAMS)
        results.append(summary)
        if h == 10:
            table10 = table
    n_shots = sum(len(load_shots(g)) for g in GAMES)
    validation_chart(results, horizons, f"Predicting shots from a single frame: {len(GAMES)} matches, {n_shots} shots",
                     OUT / "validation_auc.png")
    print("saved validation_auc.png")

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

    # 6. external test on SkillCorner (only if its tracking files are present)
    sc_ids = downloaded_skillcorner()
    if sc_ids:
        ext_results = [_external_summary(h, sc_ids, grid) for h in horizons]
        n_ext = sum(len(load_skillcorner_shots(m)) for m in sc_ids)
        validation_chart(ext_results, horizons,
                         f"External test, nothing refitted: {len(sc_ids)} SkillCorner matches, {n_ext} shots",
                         OUT / "external_auc.png")
        print("saved external_auc.png")
        bins = detection_bins(external_labelled_table(sc_ids, grid, TUNED_PARAMS, horizon_s=10))
        detection_chart(bins, "External test: AUC by tracking quality", OUT / "external_by_detection.png")
        print("saved external_by_detection.png")


if __name__ == "__main__":
    main()