"""Graph attention network vs the hand-built scores, cross-validated by match on the Metrica + DFL matches.
Same folds as the parameter tuning, same frames as validate_all.py.

python scripts/gnn_cv.py [EPOCHS]      (default 12)"""
import sys

from pitchxt import fit_xt
from pitchxt.data import load_actions
from pitchxt.dfl import downloaded_matches
from pitchxt.gnn import cross_validate_gnn
from pitchxt.gnn_data import build_graph_data, precompute_graph_data
from pitchxt.validate import (TUNED_PARAMS, _summarise, paired_summary, precompute_tuned, run_validation)

if __name__ == "__main__":             # required: data building uses worker processes
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    grid = fit_xt(load_actions(43, 106))
    games = [1, 2] + downloaded_matches()
    print(f"{len(games)} matches, {epochs} epochs per fold")

    precompute_graph_data(games)
    precompute_tuned(games, grid, TUNED_PARAMS)
    datasets = [build_graph_data(g) for g in games]

    oof = cross_validate_gnn(datasets, [str(g) for g in games], k=3, epochs=epochs)

    for h in (5, 10, 15):
        table, _ = run_validation(games, grid, horizon_s=h, tuned=TUNED_PARAMS)
        n0 = len(table)
        table = table.merge(oof[["game", "frame", "team", "gnn"]], on=["game", "frame", "team"], how="inner")
        print(f"\n=== shot within {h} s: {len(table)} frames ({n0 - len(table)} of {n0} lost in the merge), "
              f"{int(table.label.sum())} positive ===")
        print(_summarise(table).round(3).to_string(index=False))
        if h == 10:
            print("\npaired differences (10 s):")
            print(paired_summary(table).round(3).to_string(index=False))