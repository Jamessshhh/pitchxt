"""Does tracking quality explain the look-ahead model's smaller lead on well-tracked frames?
Uses the cached tables from external_test.py: python scripts/external_quality.py"""
from pitchxt import fit_xt
from pitchxt.data import load_actions
from pitchxt.external import detection_bins, external_labelled_table, paired_by_detection
from pitchxt.skillcorner import downloaded_skillcorner
from pitchxt.validate import TUNED_PARAMS

if __name__ == "__main__":
    grid = fit_xt(load_actions(43, 106))
    ids = downloaded_skillcorner()
    table = external_labelled_table(ids, grid, TUNED_PARAMS, horizon_s=10)

    print("AUC for 'shot within 10 s', by how many players were actually seen on screen:")
    print(detection_bins(table).round(3).to_string(index=False))

    print("\npaired differences on well-tracked frames only (95% intervals, 60-second blocks):")
    print(paired_by_detection(table).round(3).to_string(index=False))