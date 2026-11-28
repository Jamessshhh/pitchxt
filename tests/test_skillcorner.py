import json

import numpy as np
import pandas as pd
import pytest

from pitchxt.skillcorner import _parse_tracking, downloaded_skillcorner, load_skillcorner_match


def _write_match(tmp_path, match_id=7, use_trackable=True, lfs=False):
    d = tmp_path / "matches" / str(match_id)
    d.mkdir(parents=True)
    players = [dict(id=100 + i, trackable_object=200 + i, number=i + 1, team_id=1 if i < 2 else 2) for i in range(4)]
    match = dict(home_team=dict(id=1), away_team=dict(id=2), players=players, pitch_length=105, pitch_width=68,
                 home_team_side=["left_to_right", "right_to_left"])
    (d / f"{match_id}_match.json").write_text(json.dumps(match))
    pid = lambda i: (200 if use_trackable else 100) + i
    lines = []
    for f in (10, 11, 12, 13, 14, 15):                      # frames 14 and 15 fall outside the ball-in-play phase
        lines.append(json.dumps(dict(frame=f, period=1, ball_data=dict(x=f - 10.0, y=1.0), player_data=[
            dict(player_id=pid(0), x=-20.0 + f, y=5.0, is_detected=True),
            dict(player_id=pid(2), x=20.0, y=-5.0, is_detected=(f % 2 == 0)),
            dict(player_id=9999, x=0.0, y=0.0, is_detected=True)])))     # an unknown object (e.g. a referee)
    lines.append(json.dumps(dict(frame=3, period=None, ball_data=None, player_data=[])))   # before kick-off
    body = "x" * 10 if lfs else "\n".join(lines) + "\n"
    (d / f"{match_id}_tracking_extrapolated.jsonl").write_text(body)
    pd.DataFrame(dict(event_type=["player_possession", "player_possession", "on_ball_engagement"],
                      end_type=["shot", "pass", None], team_id=[2, 1, 1], frame_end=[13, 12, 11], period=[1, 1, 1],
                      lead_to_goal=[True, False, None], x_end=[40.0, 0.0, 0.0], y_end=[2.0, 0.0, 0.0])
                 ).to_csv(d / f"{match_id}_dynamic_events.csv", index=False)
    pd.DataFrame(dict(frame_start=[10], frame_end=[13])).to_csv(d / f"{match_id}_phases_of_play.csv", index=False)
    return tmp_path


def test_loader_maps_players_flags_ball_in_play_and_finds_shots(tmp_path, monkeypatch):
    import pitchxt.skillcorner as sc

    data = _write_match(tmp_path)
    monkeypatch.setattr(sc, "MIN_TRACKING_BYTES", 100)      # the test files are tiny
    tracking, shots = load_skillcorner_match(7, data)

    assert tracking.index.tolist() == [10, 11, 12, 13, 14, 15] and tracking.index.name == "Frame"
    assert tracking.attrs["id_key"] == "trackable_object"
    assert {"home_1_x", "home_2_x", "away_3_x", "away_4_x"} == {c for c in tracking.columns if c.endswith("_x") and c != "ball_x"}
    assert np.allclose(tracking["home_1_x"], [-10, -9, -8, -7, -6, -5])
    assert tracking["ball_alive"].tolist() == [True, True, True, True, False, False]
    assert tracking["n_detected"].tolist() == [2, 1, 2, 1, 2, 1]          # home_1 always, away_3 on even frames
    assert tracking["home_2_x"].isna().all()                                # never appears in the tracking
    assert shots["team"].tolist() == ["away"] and shots["start_frame"].tolist() == [13] and shots["goal"].tolist() == [True]


def test_loader_also_works_when_tracking_uses_the_plain_player_id(tmp_path, monkeypatch):
    import pitchxt.skillcorner as sc

    data = _write_match(tmp_path, use_trackable=False)
    monkeypatch.setattr(sc, "MIN_TRACKING_BYTES", 100)      # the test files are tiny
    tracking, _ = load_skillcorner_match(7, data)
    assert tracking.attrs["id_key"] == "id" and np.allclose(tracking["home_1_x"][:2], [-10, -9])


def test_lfs_pointers_are_not_treated_as_downloaded_data(tmp_path, monkeypatch):
    import pitchxt.skillcorner as sc

    monkeypatch.setattr(sc, "MIN_TRACKING_BYTES", 100)
    data = _write_match(tmp_path, lfs=True)
    assert downloaded_skillcorner(data) == []
    with pytest.raises(RuntimeError, match="LFS"):
        load_skillcorner_match(7, data)



def test_detection_flags_are_kept_and_detected_only_blanks_unseen_players(tmp_path, monkeypatch):
    import pitchxt.skillcorner as sc

    data = _write_match(tmp_path)
    monkeypatch.setattr(sc, "MIN_TRACKING_BYTES", 100)
    tracking, _ = load_skillcorner_match(7, data)
    assert tracking["home_1_det"].all() and tracking["away_3_det"].tolist() == [True, False, True, False, True, False]
    assert not tracking["away_3_x"].isna().any()                      # extrapolated positions are present by default

    strict, _ = load_skillcorner_match(7, data, detected_only=True)
    assert strict["away_3_x"].isna().tolist() == [False, True, False, True, False, True]    # unseen frames are blanked
    assert strict["home_1_x"].notna().all() and strict["n_detected"].tolist() == tracking["n_detected"].tolist()