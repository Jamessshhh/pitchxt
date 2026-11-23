import pandas as pd

from pitchxt.events import get_shots


def test_get_shots_filters_and_flags_goals():
    ev = pd.DataFrame({
        "type": ["PASS", "SHOT", "SHOT", "SHOT"],
        "subtype": [None, "ON TARGET-GOAL", "OFF TARGET-OUT", "HEAD-ON TARGET-GOAL"],
        "team": ["home", "home", "away", "away"],
        "start_frame": [1, 2, 3, 4],
    })
    s = get_shots(ev)
    assert len(s) == 3
    assert s["goal"].tolist() == [True, False, True]