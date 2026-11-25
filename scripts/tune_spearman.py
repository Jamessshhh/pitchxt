"""Tune the Spearman parameters with match-level cross-validation:
python scripts/tune_spearman.py [N_TRIALS] [STEP_FRAMES]"""
import sys

import pandas as pd

from pitchxt import fit_xt
from pitchxt.data import load_actions
from pitchxt.dfl import downloaded_matches
from pitchxt.tune import build_cache, cross_validate, pooled_auc, run_search, sample_trials

if __name__ == "__main__":             # required: the search uses worker processes
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 75

    grid = fit_xt(load_actions(43, 106))
    games = [1, 2] + downloaded_matches()
    print(f"matches: {games}; {n_trials} trials; one frame every {step / 25:.0f} s")

    print("preparing frames (slow the first time, cached afterwards)...")
    caches = [build_cache(g, step=step) for g in games]
    print("frames per match:", [len(c["frames"]) for c in caches])

    trials = sample_trials(n_trials)
    print("searching...")
    shares = run_search(caches, grid, trials)

    everyone = list(range(len(caches)))
    table = pd.DataFrame(trials)
    table["auc_all_matches"] = [pooled_auc(s, caches, everyone) for s in shares]
    table.to_csv("data/processed/tuning_trials.csv", index=False)
    print("\ndefaults (trial 0):", round(table.auc_all_matches[0], 4))
    print("\nbest five trials, in-sample (the same matches they were chosen on):")
    print(table.sort_values("auc_all_matches", ascending=False).head(5).round(3).to_string())
    print(f"\nspread across all trials: min {table.auc_all_matches.min():.3f}, "
          f"median {table.auc_all_matches.median():.3f}, max {table.auc_all_matches.max():.3f}")

    folds, overall = cross_validate(shares, caches, k=3)
    print("\n=== held-out evaluation (parameters chosen without the test matches) ===")
    print(folds.round(4).to_string(index=False))
    print(f"\npooled held-out AUC: default {overall['auc_default']:.4f}, tuned {overall['auc_tuned']:.4f}")
    print(f"difference (tuned - default): {overall['diff']:+.4f} "
          f"(95% interval {overall['ci_low']:+.4f} to {overall['ci_high']:+.4f})")
    for i, row in folds.iterrows():
        print(f"fold {i} chose:", {k: round(v, 2) for k, v in trials[int(row.best_trial)].items()})