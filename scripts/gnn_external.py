"""Train the graph network on the Metrica + DFL matches, then test it, frozen, on the SkillCorner matches.

python scripts/gnn_external.py [EPOCHS]      (default 12)"""
import sys

import pandas as pd
import torch

from pitchxt import fit_xt
from pitchxt.data import load_actions
from pitchxt.dfl import downloaded_matches
from pitchxt.external import (BLOCK_FRAMES, detection_bins, external_labelled_table, paired_external,
                              precompute_external)
from pitchxt.gnn import _stack, make_rows, predict, prediction_table, train_model
from pitchxt.gnn_data import build_graph_data, build_graph_data_sc, precompute_graph_data
from pitchxt.skillcorner import downloaded_skillcorner
from pitchxt.validate import TUNED_PARAMS, _summarise

if __name__ == "__main__":             # required: data building uses worker processes
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    grid = fit_xt(load_actions(43, 106))
    games = [1, 2] + downloaded_matches()
    sc_ids = downloaded_skillcorner()
    print(f"train on {len(games)} matches, test on {len(sc_ids)} SkillCorner matches; {epochs} epochs, nothing tuned on the test set")

    precompute_graph_data(games, sc_ids=sc_ids)
    precompute_external(sc_ids, grid, TUNED_PARAMS)

    rows = [make_rows(build_graph_data(g)) for g in games]
    model, _ = train_model(*_stack(rows, range(len(rows))), epochs=epochs)
    torch.save(model.state_dict(), "data/processed/gnn_model.pt")

    preds = []
    for m in sc_ids:
        nodes, mask, _, meta = make_rows(build_graph_data_sc(m))
        preds.append(prediction_table(m, meta, predict(model, nodes, mask)))
    preds = pd.concat(preds, ignore_index=True)

    for h in (5, 10, 15):
        table = external_labelled_table(sc_ids, grid, TUNED_PARAMS, horizon_s=h)
        n0 = len(table)
        table = table.merge(preds[["game", "frame", "team", "gnn"]], on=["game", "frame", "team"], how="inner")
        print(f"\n=== external, shot within {h} s: {len(table)} frames ({n0 - len(table)} of {n0} lost in the merge), "
              f"{int(table.label.sum())} positive ===")
        print(_summarise(table, block_frames=BLOCK_FRAMES).round(3).to_string(index=False))
        if h == 10:
            print("\npaired differences (10 s):")
            print(paired_external(table).round(3).to_string(index=False))
            print("\nAUC by how many players were actually detected:")
            print(detection_bins(table).round(3).to_string(index=False))