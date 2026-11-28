"""A graph attention network over the 22 players and the ball.

Each sampled frame becomes two graphs, one per team's point of view (always attacking toward +x).
Nodes are players and the ball with position, velocity and a team flag; every pair of nodes
exchanges messages through attention that also sees relative position, distance and relative
velocity. The output is the probability that the viewing team shoots within the next N seconds.
The model does not depend on the order of the players, and padded (absent) players are ignored.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .validate import auc

POS_SCALE = np.array([52.5, 34.0], dtype=np.float32)    # metres -> roughly [-1, 1]
VEL_SCALE = 5.0                                          # m/s
NODE_FEATURES = 7                                        # x, y, vx, vy, is_own, is_opponent, is_ball
EDGE_FEATURES = 6                                        # dx, dy, distance, dvx, dvy, same_team


def make_rows(data, horizon=10):
    """Two graphs per frame (home's view first, then away's), as arrays.

    Returns nodes (2n, 2P+1, 7), mask (2n, 2P+1), labels (2n,) and a meta DataFrame (frame, team,
    n_detected). Node order: own players, opponent players, ball (always last). The away view is the
    home view rotated by 180 degrees, so the viewing team always attacks toward +x."""
    n, _, P, _ = data["players"].shape
    nodes = np.zeros((2 * n, 2 * P + 1, NODE_FEATURES), dtype=np.float32)
    mask = np.zeros((2 * n, 2 * P + 1), dtype=bool)
    for view, (own, opp) in enumerate(((0, 1), (1, 0))):
        sl = slice(view * n, (view + 1) * n)
        sign = 1.0 if view == 0 else -1.0
        for block, team, flag in ((0, own, 4), (1, opp, 5)):
            cols = slice(block * P, (block + 1) * P)
            a = data["players"][:, team] * sign
            nodes[sl, cols, 0:2] = a[..., 0:2] / POS_SCALE
            nodes[sl, cols, 2:4] = a[..., 2:4] / VEL_SCALE
            nodes[sl, cols, flag] = 1.0
            mask[sl, cols] = data["mask"][:, team]
        b = data["ball"] * sign
        nodes[sl, -1, 0:2] = b[:, 0:2] / POS_SCALE
        nodes[sl, -1, 2:4] = b[:, 2:4] / VEL_SCALE
        nodes[sl, -1, 6] = 1.0
        mask[sl, -1] = True
    nodes[~mask] = 0.0                                   # absent players carry no information at all
    y = np.concatenate([data[f"y_home_{horizon}"], data[f"y_away_{horizon}"]]).astype(np.float32)
    meta = pd.DataFrame({"frame": np.tile(data["frames"], 2), "team": ["home"] * n + ["away"] * n,
                         "n_detected": np.tile(data["n_detected"], 2)})
    return nodes, mask, y, meta


def pairwise_features(x):
    """(B, N, N, 6) relative position (in 20 m units), distance, relative velocity, same-team flag."""
    pos = x[..., 0:2] * torch.as_tensor(POS_SCALE, device=x.device) / 20.0
    vel = x[..., 2:4]
    d = pos[:, :, None, :] - pos[:, None, :, :]
    dist = d.norm(dim=-1, keepdim=True)
    dv = vel[:, :, None, :] - vel[:, None, :, :]
    own, opp = x[..., 4], x[..., 5]
    same = (own[:, :, None] * own[:, None, :] + opp[:, :, None] * opp[:, None, :]).unsqueeze(-1)
    return torch.cat([d, dist, dv, same], dim=-1)


class GraphAttentionLayer(nn.Module):
    """Multi-head attention over all node pairs, with a learned bias from the pairwise features."""

    def __init__(self, dim, heads, edge_dim, dropout):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dk = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.edge = nn.Sequential(nn.Linear(edge_dim, 32), nn.GELU(), nn.Linear(32, heads))
        self.out = nn.Linear(dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge, mask):
        B, N, D = h.shape
        q, k, v = self.qkv(h).view(B, N, 3, self.heads, self.dk).permute(2, 0, 3, 1, 4)    # (B, H, N, dk) each
        logits = q @ k.transpose(-1, -2) / self.dk ** 0.5 + self.edge(edge).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~mask[:, None, None, :], float("-inf"))               # never attend to absent players
        attn = self.drop(torch.softmax(logits, dim=-1))
        z = (attn @ v).transpose(1, 2).reshape(B, N, D)
        h = self.norm1(h + self.drop(self.out(z)))
        return self.norm2(h + self.drop(self.ff(h)))


class PitchGAT(nn.Module):
    def __init__(self, dim=64, heads=4, layers=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(NODE_FEATURES, dim), nn.GELU(), nn.Linear(dim, dim))
        self.layers = nn.ModuleList(GraphAttentionLayer(dim, heads, EDGE_FEATURES, dropout) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, x, mask):
        """x: (B, N, 7) with the ball as the last node; mask: (B, N) bool. Returns logits (B,)."""
        edge = pairwise_features(x)
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, edge, mask)
        players = mask[:, :-1].unsqueeze(-1).to(h.dtype)
        mean = (h[:, :-1] * players).sum(1) / players.sum(1).clamp(min=1.0)
        return self.head(torch.cat([h[:, -1], mean], dim=-1)).squeeze(-1)


def _flip_y(x, flip):
    """Mirror the pitch across its long axis (y and vy change sign) for the rows where `flip` is True."""
    x = x.clone()
    sign = 1.0 - 2.0 * flip.to(x.dtype)[:, None]
    x[..., 1] *= sign
    x[..., 3] *= sign
    return x


@torch.no_grad()
def predict(model, nodes, mask, device="cpu", batch_size=2048, mirror=True):
    """Probabilities, averaged over the pitch and its mirror image (the game is symmetric in y)."""
    model.eval()
    out = []
    for i in range(0, len(nodes), batch_size):
        xb = torch.from_numpy(nodes[i:i + batch_size]).to(device)
        mb = torch.from_numpy(mask[i:i + batch_size]).to(device)
        p = torch.sigmoid(model(xb, mb))
        if mirror:
            p = 0.5 * (p + torch.sigmoid(model(_flip_y(xb, torch.ones(len(xb), dtype=torch.bool, device=device)), mb)))
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def train_model(nodes, mask, y, epochs=12, batch_size=256, lr=2e-3, weight_decay=1e-2, seed=0,
                device="cpu", eval_data=None, verbose=True, player_dropout=0.0, **model_kw):
    """Fixed-budget training. eval_data=(nodes, mask, y) is only for printing a diagnostic curve; it never
    influences training or any choice made here. player_dropout hides random players during training."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PitchGAT(**model_kw).to(device)
    X, M, Y = torch.from_numpy(nodes), torch.from_numpy(mask), torch.from_numpy(y)
    n = len(Y)
    steps = max(1, int(np.ceil(n / batch_size)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(n))
        total = 0.0
        for i in range(steps):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            xb, mb, yb = X[idx].to(device), M[idx].to(device), Y[idx].to(device)
            xb = _flip_y(xb, torch.rand(len(idx), device=device) < 0.5)           # augmentation
            if player_dropout > 0:                                                  # hide random players (never the ball)
                hide = torch.rand(mb.shape, device=device) < player_dropout
                hide[:, -1] = False
                mb = mb & ~hide
                xb = xb.masked_fill(~mb.unsqueeze(-1), 0.0)
            loss = F.binary_cross_entropy_with_logits(model(xb, mb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        entry = {"epoch": epoch + 1, "train_loss": total / n}
        if eval_data is not None:
            entry["eval_auc"] = auc(predict(model, eval_data[0], eval_data[1], device), eval_data[2])
        history.append(entry)
        if verbose:
            print("   ", {k: round(v, 4) if isinstance(v, float) else v for k, v in entry.items()}, flush=True)
    return model, history


def _stack(rows, ids):
    return (np.concatenate([rows[m][0] for m in ids]), np.concatenate([rows[m][1] for m in ids]),
            np.concatenate([rows[m][2] for m in ids]))


def prediction_table(game, meta, p):
    out = meta.copy()
    out.insert(0, "game", str(game))
    out["gnn"] = p
    return out


def cross_validate_gnn(datasets, names, k=3, seed=0, horizon=10, device="cpu", **train_kw):
    """Match-level k-fold with the same fold assignment as tune.cross_validate (same seed, same order).
    Returns a table of out-of-fold predictions: game, frame, team, n_detected, gnn."""
    rows = [make_rows(d, horizon) for d in datasets]
    n_m = len(datasets)
    k = min(k, n_m)
    order = np.random.default_rng(seed).permutation(n_m)
    folds = [sorted(order[i::k].tolist()) for i in range(k)]
    parts = []
    for fi, test in enumerate(folds):
        train = [m for m in range(n_m) if m not in test]
        print(f"fold {fi}: train on {[names[m] for m in train]}, test on {[names[m] for m in test]}", flush=True)
        model, _ = train_model(*_stack(rows, train)[:3], device=device, eval_data=_stack(rows, test), seed=seed + fi, **train_kw)
        for m in test:
            parts.append(prediction_table(names[m], rows[m][3], predict(model, rows[m][0], rows[m][1], device)))
    return pd.concat(parts, ignore_index=True)