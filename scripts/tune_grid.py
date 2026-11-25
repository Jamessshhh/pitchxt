"""Two-parameter grids over the Spearman model (all other parameters at their defaults):

python scripts/tune_grid.py                    # reaction time x ball speed (42 settings)
python scripts/tune_grid.py lookahead          # look-ahead x reaction delay (35 settings)
python scripts/tune_grid.py lookahead quick    # tiny version, to check everything runs"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitchxt import fit_xt
from pitchxt.data import load_actions
from pitchxt.dfl import downloaded_matches
from pitchxt.tune import build_cache, cross_validate, grid_trials_for, pooled_auc, run_search

# mode: (row parameter, row values, row label, column parameter, column values, column label, image name)
MODES = {
    "reaction_ball": ("reaction_time", [0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0], "reaction time (s)",
                      "average_ball_speed", [12, 15, 20, 25, 30, 40], "average ball speed (m/s)",
                      "tuning_grid.png"),
    "lookahead": ("lookahead", [0.0, 0.7, 1.5, 2.0, 3.0, 4.0, 5.0], "look-ahead (s)",
                  "reaction_time", [0.0, 0.3, 0.7, 1.0, 1.5], "reaction delay (s)",
                  "tuning_lookahead.png"),
}
DEFAULT_CELL = {"reaction_ball": (0.7, 15), "lookahead": (0.7, 0.7)}   # the published defaults

if __name__ == "__main__":             # required: the search uses worker processes
    args = sys.argv[1:]
    mode = "lookahead" if "lookahead" in args else "reaction_ball"
    quick = "quick" in args
    rp, rv, rl, cp, cv, cl, image = MODES[mode]
    if quick:
        rv, cv = rv[:2], cv[:2]

    grid = fit_xt(load_actions(43, 106))
    games = [1, 2] + downloaded_matches()
    print(f"mode {mode}: matches {games}; grid {len(rv)} x {len(cv)} = {len(rv) * len(cv)} settings (+ the defaults)")
    caches = [build_cache(g, step=75) for g in games]

    trials = grid_trials_for({rp: rv, cp: cv})
    shares = run_search(caches, grid, trials)

    everyone = list(range(len(caches)))
    aucs = np.array([pooled_auc(s, caches, everyone) for s in shares])
    pd.DataFrame(trials).assign(auc_all_matches=aucs).to_csv(f"data/processed/tuning_{mode}.csv", index=False)

    surface = aucs[1:].reshape(len(rv), len(cv))
    print(f"\ndefaults: {aucs[0]:.4f}")
    print(f"\nin-sample AUC, rows = {rl}, columns = {cl}:")
    print(pd.DataFrame(surface, index=rv, columns=cv).round(4).to_string())
    bi, bj = np.unravel_index(np.argmax(surface), surface.shape)
    print(f"\nbest: {rl} {rv[bi]}, {cl} {cv[bj]} -> {surface[bi, bj]:.4f}")
    edges = [name for name, hit in ((rl, bi in (0, len(rv) - 1)), (cl, bj in (0, len(cv) - 1))) if hit]
    print("the best point is on the edge of the grid for:", ", ".join(edges) if edges else "neither parameter (good)")

    folds, overall = cross_validate(shares, caches, k=3)
    print("\n=== held-out evaluation (setting chosen without the test matches) ===")
    print(folds.round(4).to_string(index=False))
    print(f"\npooled held-out AUC: default {overall['auc_default']:.4f}, tuned {overall['auc_tuned']:.4f}")
    print(f"difference (tuned - default): {overall['diff']:+.4f} "
          f"(95% interval {overall['ci_low']:+.4f} to {overall['ci_high']:+.4f})")
    for i, row in folds.iterrows():
        t = trials[int(row.best_trial)]
        shown = lambda v: "the default" if v is None else v
        print(f"fold {i} chose: {rl} {shown(t.get(rp))}, {cl} {shown(t.get(cp))}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(surface, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cv)), [str(c) for c in cv])
    ax.set_yticks(range(len(rv)), [str(r) for r in rv])
    ax.set_xlabel(cl)
    ax.set_ylabel(rl)
    for i in range(len(rv)):
        for j in range(len(cv)):
            ax.text(j, i, f"{surface[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if surface[i, j] < surface.max() - 0.004 else "black")
    ax.scatter([bj], [bi], marker="*", s=220, c="#d6455d", edgecolors="white", zorder=3, label="best")
    dr, dc = DEFAULT_CELL[mode]
    if dr in rv and dc in cv:
        ax.scatter([cv.index(dc)], [rv.index(dr)], marker="s", s=140, facecolors="none",
                   edgecolors="white", linewidths=2, zorder=3, label="published defaults")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Shot-prediction AUC across Spearman parameters")
    plt.colorbar(im, ax=ax, label="AUC (all matches, in-sample)")
    fig.savefig(f"docs/{image}", dpi=130, bbox_inches="tight")
    print(f"\nsaved docs/{image}")