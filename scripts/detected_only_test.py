"""Is the external result an artefact of SkillCorner's extrapolated players?

Everything is rerun using ONLY the players that SkillCorner actually saw on screen in each frame (the
others are removed), for the hand-built scores and for the graph network alike. A second network is
trained with random player dropout so that it copes with missing players fairly.

python scripts/detected_only_test.py [EPOCHS]      (default 12)"""
import sys
from pathlib import Path

import pandas as pd
import torch

from pitchxt import fit_xt
from pitchxt.data import load_actions
from pitchxt.dfl import downloaded_matches
from pitchxt.external import (BLOCK_FRAMES, detection_bins, external_labelled_table, paired_external,
                              precompute_external)
from pitchxt.gnn import PitchGAT, _stack, make_rows, predict, prediction_table, train_model
from pitchxt.gnn_data import build_graph_data, build_graph_data_sc, precompute_graph_data
from pitchxt.skillcorner import downloaded_skillcorner
from pitchxt.validate import TUNED_PARAMS, _summarise, paired_bootstrap

if __name__ == "__main__":             # required: data building uses worker processes
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    grid = fit_xt(load_actions(43, 106))
    games = [1, 2] + downloaded_matches()
    sc_ids = downloaded_skillcorner()
    print(f"{len(sc_ids)} SkillCorner matches, only players actually detected on screen; {epochs} epochs")

    precompute_graph_data(games, sc_ids=sc_ids, detected_only=True)
    precompute_external(sc_ids, grid, TUNED_PARAMS, detected_only=True)

    rows = [make_rows(build_graph_data(g)) for g in games]
    train = _stack(rows, range(len(rows)))

    models = {}
    path_a = Path("data/processed/gnn_model.pt")
    if path_a.exists():                                   # the network from gnn_external.py (no dropout)
        a = PitchGAT()
        a.load_state_dict(torch.load(path_a))
        models["gnn"] = a
    else:
        models["gnn"], _ = train_model(*train, epochs=epochs)
    print("training the player-dropout network...")
    models["gnn_dropout"], _ = train_model(*train, epochs=epochs, player_dropout=0.3)
    torch.save(models["gnn_dropout"].state_dict(), "data/processed/gnn_model_dropout.pt")

    preds = None
    for name, model in models.items():
        parts = []
        for m in sc_ids:
            nodes, mask, _, meta = make_rows(build_graph_data_sc(m, detected_only=True))
            parts.append(prediction_table(m, meta, predict(model, nodes, mask)).rename(columns={"gnn": name}))
        p = pd.concat(parts, ignore_index=True)[["game", "frame", "team", name]]
        preds = p if preds is None else preds.merge(p, on=["game", "frame", "team"])

    for h in (5, 10, 15):
        table = external_labelled_table(sc_ids, grid, TUNED_PARAMS, horizon_s=h, detected_only=True)
        n0 = len(table)
        table = table.merge(preds, on=["game", "frame", "team"], how="inner")
        print(f"\n=== detected players only, shot within {h} s: {len(table)} frames "
              f"({n0 - len(table)} of {n0} lost in the merge), {int(table.label.sum())} positive ===")
        print(_summarise(table, block_frames=BLOCK_FRAMES).round(3).to_string(index=False))
        if h == 10:
            print("\npaired differences (10 s):")
            print(paired_external(table).round(3).to_string(index=False))
            rows_ = []
            for a, b, label in (("gnn_dropout", "share_tuned", "Graph network (dropout) vs Spearman + look-ahead"),
                                ("gnn_dropout", "depth", "Graph network (dropout) vs team depth"),
                                ("gnn_dropout", "gnn", "Graph network (dropout) vs graph network")):
                d, lo, hi, _ = paired_bootstrap(table, a, b, n_boot=500, block_frames=BLOCK_FRAMES)
                rows_.append(dict(comparison=label, auc_diff=d, ci_low=lo, ci_high=hi))
            print(pd.DataFrame(rows_).round(3).to_string(index=False))
            print("\nAUC by how many players were actually detected:")
            print(detection_bins(table).round(3).to_string(index=False))