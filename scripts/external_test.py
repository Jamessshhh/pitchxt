"""External test on the SkillCorner open data, with nothing refitted:
python scripts/external_test.py [WORKERS]       (default 4; use 2 if memory gets tight)

Setup (the tracking files are Git LFS objects, ~1.8 GB for all 20 matches):
    brew install git-lfs && git lfs install
    git clone --depth 1 https://github.com/SkillCorner/opendata.git data/raw/skillcorner"""
import sys

from pitchxt import fit_xt
from pitchxt.data import load_actions
from pitchxt.external import paired_external, precompute_external, run_external
from pitchxt.skillcorner import downloaded_skillcorner, load_skillcorner_shots
from pitchxt.validate import TUNED_PARAMS

if __name__ == "__main__":             # required: the precompute step uses worker processes
    grid = fit_xt(load_actions(43, 106))
    ids = downloaded_skillcorner()
    if not ids:
        raise SystemExit("No SkillCorner tracking found in data/raw/skillcorner/data/matches. "
                         "Did git-lfs pull the real files? (see the setup note at the top of this script)")
    print(f"{len(ids)} matches, {sum(len(load_skillcorner_shots(m)) for m in ids)} shots")
    print("parameters:", TUNED_PARAMS, "(chosen on the Metrica and DFL matches, not refitted here)")

    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    precompute_external(ids, grid, TUNED_PARAMS, workers=workers)

    for h in (5, 10, 15):
        table, summary = run_external(ids, grid, TUNED_PARAMS, horizon_s=h)
        if h == 5:
            nd = table.drop_duplicates(["game", "frame"])["n_detected"]
            print(f"\nplayers detected on screen per frame: median {nd.median():.0f}, "
                  f"{(nd >= 16).mean():.0%} of frames have at least 16")
        print(f"\n=== shot within {h} s: {len(table)} sampled frames, {int(table.label.sum())} positive ===")
        print(summary.round(3).to_string(index=False))
        if h == 10:
            print("\npaired differences (10 s):")
            print(paired_external(table).round(3).to_string(index=False))
            for k in (12, 16):
                sub, strict = run_external(ids, grid, TUNED_PARAMS, horizon_s=10, min_detected=k)
                print(f"\nonly frames where at least {k} players are detected "
                      f"({len(sub)} frames, {int(sub.label.sum())} positive):")
                print(strict.round(3).to_string(index=False))