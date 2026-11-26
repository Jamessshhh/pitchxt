"""Validation on every available match, including the tuned look-ahead model:
python scripts/validate_all.py"""
from pitchxt import fit_xt
from pitchxt.data import load_actions
from pitchxt.dfl import downloaded_matches
from pitchxt.matches import load_shots
from pitchxt.validate import (TUNED_PARAMS, paired_summary, precompute_tuned, run_validation,
                              summary_by_source)

if __name__ == "__main__":             # required: the precompute step uses worker processes
    grid = fit_xt(load_actions(43, 106))
    games = [1, 2] + downloaded_matches()
    print("matches:", games)
    print("shots:", sum(len(load_shots(g)) for g in games))
    print("tuned parameters:", TUNED_PARAMS)

    precompute_tuned(games, grid, TUNED_PARAMS)

    for h in (5, 10, 15):
        table, summary = run_validation(games, grid, horizon_s=h, tuned=TUNED_PARAMS)
        print(f"\n=== shot within {h} s: {len(table)} sampled frames, {int(table.label.sum())} positive ===")
        print(summary.round(3).to_string(index=False))
        print("\nby source:")
        print(summary_by_source(table).round(3).to_string(index=False))
        if h == 10:
            print("\npaired differences (10 s):")
            print(paired_summary(table).round(3).to_string(index=False))