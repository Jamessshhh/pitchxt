import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pitchxt.gnn import (NODE_FEATURES, PitchGAT, cross_validate_gnn, make_rows, pairwise_features, predict,
                         train_model)
from pitchxt.gnn_data import MAX_PER_TEAM


def _data(n=200, seed=0, signal=True):
    """Synthetic frames where the home team 'shoots soon' when its players are pushed up the pitch."""
    rng = np.random.default_rng(seed)
    P = MAX_PER_TEAM
    players = np.zeros((n, 2, P, 4), dtype=np.float32)
    mask = np.zeros((n, 2, P), dtype=bool)
    mask[:, :, :10] = True
    push = rng.random(n) < 0.3
    players[:, 0, :, 0] = rng.normal(-5, 8, (n, P)) + 15 * push[:, None]     # home x
    players[:, 1, :, 0] = rng.normal(5, 8, (n, P))                            # away x
    players[:, :, :, 1] = rng.normal(0, 15, (n, 2, P))
    players[:, :, :, 2:4] = rng.normal(0, 1.5, (n, 2, P, 2))
    players *= mask[..., None]
    ball = np.concatenate([rng.normal(0, 15, (n, 2)), rng.normal(0, 3, (n, 2))], axis=1).astype(np.float32)
    frames = np.arange(n) * 25 + 1000
    label = push if signal else rng.random(n) < 0.3
    return dict(players=players, mask=mask, ball=ball, frames=frames, n_detected=np.full(n, 20),
                y_home_10=label, y_away_10=np.zeros(n, dtype=bool))


def test_rows_are_oriented_from_each_teams_own_point_of_view():
    d = _data(8)
    nodes, mask, y, meta = make_rows(d)
    n = 8
    assert nodes.shape == (2 * n, 2 * MAX_PER_TEAM + 1, NODE_FEATURES) and meta["team"].tolist() == ["home"] * n + ["away"] * n
    # the home view's own players are the home players, the away view's own players are the away players, rotated 180 degrees
    assert np.allclose(nodes[0, 0, 0] * 52.5, d["players"][0, 0, 0, 0])
    assert np.allclose(nodes[n, 0, 0] * 52.5, -d["players"][0, 1, 0, 0])
    assert nodes[0, 0, 4] == 1 and nodes[0, MAX_PER_TEAM, 5] == 1 and nodes[0, -1, 6] == 1
    assert (nodes[~mask] == 0).all()                                      # absent players are all zeros


def test_output_does_not_depend_on_the_order_of_the_players():
    torch.manual_seed(0)
    nodes, mask, _, _ = make_rows(_data(16))
    model = PitchGAT().eval()
    x, m = torch.from_numpy(nodes), torch.from_numpy(mask)
    perm = torch.cat([torch.randperm(2 * MAX_PER_TEAM), torch.tensor([2 * MAX_PER_TEAM])])    # shuffle players, keep the ball last
    with torch.no_grad():
        assert torch.allclose(model(x, m), model(x[:, perm], m[:, perm]), atol=1e-5)


def test_absent_players_cannot_influence_the_output():
    torch.manual_seed(0)
    nodes, mask, _, _ = make_rows(_data(16))
    model = PitchGAT().eval()
    x, m = torch.from_numpy(nodes), torch.from_numpy(mask)
    garbage = x.clone()
    garbage[~m] = 5.0 * torch.randn_like(garbage[~m])                    # junk in the padded slots
    with torch.no_grad():
        assert torch.allclose(model(x, m), model(garbage, m), atol=1e-5)


def test_pairwise_features_have_the_expected_shape_and_diagonal():
    x = torch.from_numpy(make_rows(_data(4))[0])
    e = pairwise_features(x)
    assert e.shape == (8, 2 * MAX_PER_TEAM + 1, 2 * MAX_PER_TEAM + 1, 6)
    assert torch.allclose(e[:, range(25), range(25), :5], torch.zeros(8, 25, 5), atol=1e-6)     # a node vs itself: no offset


def test_model_learns_a_simple_pattern_and_cross_validation_runs():
    train = make_rows(_data(600, seed=1))
    test = make_rows(_data(300, seed=2))
    model, hist = train_model(train[0], train[1], train[2], epochs=6, batch_size=128, verbose=False,
                              eval_data=(test[0], test[1], test[2]))
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert hist[-1]["eval_auc"] > 0.85
    assert predict(model, test[0], test[1]).shape == (600,)

    preds = cross_validate_gnn([_data(150, seed=s) for s in (3, 4, 5)], ["a", "b", "c"], k=3, epochs=2, batch_size=64)
    assert set(preds["game"]) == {"a", "b", "c"} and preds["gnn"].between(0, 1).all()
    assert len(preds) == 2 * 3 * 150