"""Grid over reaction time and ball speed (other parameters at their defaults):
python scripts/tune_grid.py          # full grid, 42 settings
python scripts/tune_grid.py quick    # 2 x 2 grid, to check everything runs"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitchxt import fit_xt
from pitchxt.data import load_actions
from pitchxt.dfl import downloaded_matches
from pitchxt.tune import build_cache, cross_validate, grid_trials, pooled_auc, run_search

if __name__ == "__main__":             # required: the search uses worker processes
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    RT = [0.7, 1.1] if quick else [0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0]       # reaction time (s)
    BS = [15, 25] if quick else [12, 15, 20, 25, 30, 40]                     # ball speed (m/s)

    grid = fit_xt(load_actions(43, 106))
    games = [1, 2] + downloaded_matches()
    print(f"matches: {games}; grid {len(RT)} x {len(BS)} = {len(RT) * len(BS)} settings (+ the defaults)")
    caches = [build_cache(g, step=75) for g in games]

    trials = grid_trials(RT, BS)
    shares = run_search(caches, grid, trials)

    everyone = list(range(len(caches)))
    aucs = np.array([pooled_auc(s, caches, everyone) for s in shares])
    pd.DataFrame(trials).assign(auc_all_matches=aucs).to_csv("data/processed/tuning_grid.csv", index=False)

    surface = aucs[1:].reshape(len(RT), len(BS))
    print(f"\ndefaults (reaction 0.7 s, ball 15 m/s): {aucs[0]:.4f}")
    print("\nin-sample AUC, rows = reaction time (s), columns = ball speed (m/s):")
    print(pd.DataFrame(surface, index=RT, columns=BS).round(4).to_string())
    bi, bj = np.unravel_index(np.argmax(surface), surface.shape)
    print(f"\nbest: reaction {RT[bi]} s, ball {BS[bj]} m/s -> {surface[bi, bj]:.4f}")
    edges = []
    if bi in (0, len(RT) - 1):
        edges.append("reaction time")
    if bj in (0, len(BS) - 1):
        edges.append("ball speed")
    print("the best point is on the edge of the grid for:", ", ".join(edges) if edges else "neither parameter (good)")

    folds, overall = cross_validate(shares, caches, k=3)
    print("\n=== held-out evaluation (setting chosen without the test matches) ===")
    print(folds.round(4).to_string(index=False))
    print(f"\npooled held-out AUC: default {overall['auc_default']:.4f}, tuned {overall['auc_tuned']:.4f}")
    print(f"difference (tuned - default): {overall['diff']:+.4f} "
          f"(95% interval {overall['ci_low']:+.4f} to {overall['ci_high']:+.4f})")
    for i, row in folds.iterrows():
        t = trials[int(row.best_trial)]
        print(f"fold {i} chose: reaction {t['reaction_time']} s, ball {t['average_ball_speed']} m/s")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(surface, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(BS)), [str(b) for b in BS])
    ax.set_yticks(range(len(RT)), [str(r) for r in RT])
    ax.set_xlabel("average ball speed (m/s)")
    ax.set_ylabel("reaction time (s)")
    for i in range(len(RT)):
        for j in range(len(BS)):
            ax.text(j, i, f"{surface[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if surface[i, j] < surface.max() - 0.004 else "black")
    ax.scatter([bj], [bi], marker="*", s=220, c="#d6455d", edgecolors="white", zorder=3, label="best")
    if 0.7 in RT and 15 in BS:
        ax.scatter([BS.index(15)], [RT.index(0.7)], marker="s", s=140, facecolors="none",
                   edgecolors="white", linewidths=2, zorder=3, label="published defaults")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Shot-prediction AUC across Spearman parameters")
    plt.colorbar(im, ax=ax, label="AUC (all matches, in-sample)")
    fig.savefig("docs/tuning_grid.png", dpi=130, bbox_inches="tight")
    print("\nsaved docs/tuning_grid.png")