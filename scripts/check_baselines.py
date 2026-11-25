"""Does the Spearman model add anything beyond simple team-movement features?
python scripts/check_baselines.py [LOOKAHEAD] [REACTION_DELAY]      (defaults: 5.0 0.3)"""
import sys

import numpy as np
import pandas as pd

from pitchxt import fit_xt
from pitchxt.data import load_actions
from pitchxt.dfl import downloaded_matches
from pitchxt.tune import (BASELINE_NAMES, baseline_scores, build_cache, grid_trials_for, paired_auc_diff,
                          run_search, stack_rows)
from pitchxt.validate import auc

if __name__ == "__main__":             # required: the search uses worker processes
    look = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

    grid = fit_xt(load_actions(43, 106))
    games = [1, 2] + downloaded_matches()
    caches = [build_cache(g, step=75) for g in games]
    print(f"{len(games)} matches, {sum(len(c['frames']) for c in caches)} frames")

    # trial 0 = published defaults, trial 1 = look-ahead / delay from the grid search
    trials = grid_trials_for({"lookahead": [look], "reaction_time": [delay]})
    shares = run_search(caches, grid, trials)

    methods = {}
    for name, idx in (("Spearman, published defaults", 0), (f"Spearman, look-ahead {look} s", 1)):
        methods[name] = stack_rows(caches, [s for s in shares[idx]], [1 - s for s in shares[idx]])
    per_match = [baseline_scores(c, grid) for c in caches]
    for name in BASELINE_NAMES:
        methods[name] = stack_rows(caches, [p[name][0] for p in per_match], [p[name][1] for p in per_match])

    rows = [(name, auc(s, y)) for name, (s, y, _) in methods.items()]
    table = pd.DataFrame(rows, columns=["score", "auc"]).sort_values("auc", ascending=False)
    print("\nAUC for 'shot within 10 s', all matches pooled:")
    print(table.round(4).to_string(index=False))

    tuned_name = f"Spearman, look-ahead {look} s"
    best_simple = max(BASELINE_NAMES, key=lambda k: auc(methods[k][0], methods[k][1]))
    print(f"\nbest simple feature: {best_simple}")
    for name in ("Spearman, published defaults", tuned_name):
        a, y, b = methods[name]
        d, lo, hi = paired_auc_diff(a, methods[best_simple][0], y, b)
        print(f"{name} minus {best_simple}: {d:+.4f} (95% interval {lo:+.4f} to {hi:+.4f})")