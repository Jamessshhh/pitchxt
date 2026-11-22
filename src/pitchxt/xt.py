"""Markov-chain expected threat (xT), after Karun Singh. StatsBomb coordinates."""
import numpy as np

L, W = 120, 80


def _cell(x, y, nx, ny):
    """Map coordinates to (row, col) grid indices, clipped to the pitch."""
    cx = np.clip((np.asarray(x, dtype=float) / L * nx).astype(int), 0, nx - 1)
    cy = np.clip((np.asarray(y, dtype=float) / W * ny).astype(int), 0, ny - 1)
    return cy, cx


def fit_xt(actions, nx=16, ny=12, n_iter=100, tol=1e-9, prior_strength=10):
    """Fit an xT grid of shape (ny, nx) from an actions DataFrame.

    For each cell:
        xT = P(shoot) * P(goal | shot) + P(move) * sum_over_cells( P(move to c') * xT[c'] )
    solved by iterating until it converges. Failed moves lose possession,
    so they contribute nothing to the second term.

    prior_strength shrinks each cell's goal rate toward the overall goal rate
    per shot, acting like that many extra "average" shots. It stops cells with
    only a few shots from getting extreme values. Set to 0 to disable.
    """
    n = nx * ny
    cy0, cx0 = _cell(actions["x0"], actions["y0"], nx, ny)
    start = cy0 * nx + cx0

    is_shot = (actions["kind"] == "shot").to_numpy()
    is_move = ~is_shot
    is_goal = actions["goal"].to_numpy()
    is_ok = actions["success"].to_numpy()

    shots = np.bincount(start[is_shot], minlength=n).astype(float)
    goals = np.bincount(start[is_shot & is_goal], minlength=n).astype(float)
    moves = np.bincount(start[is_move], minlength=n).astype(float)
    total = shots + moves

    with np.errstate(divide="ignore", invalid="ignore"):
        p_shot = np.where(total > 0, shots / total, 0.0)
        p_move = np.where(total > 0, moves / total, 0.0)

    # shrink small-sample cells toward the overall goal rate per shot
    global_rate = goals.sum() / max(shots.sum(), 1)
    denom = shots + prior_strength
    with np.errstate(divide="ignore", invalid="ignore"):
        p_goal = np.where(denom > 0, (goals + prior_strength * global_rate) / denom, 0.0)

    # transition matrix from successful moves, normalised by all moves from that cell
    ok = is_move & is_ok
    cy1, cx1 = _cell(actions["x1"].to_numpy()[ok], actions["y1"].to_numpy()[ok], nx, ny)
    end = cy1 * nx + cx1
    T = np.zeros((n, n))
    np.add.at(T, (start[ok], end), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(moves[:, None] > 0, T / moves[:, None], 0.0)

    xt = np.zeros(n)
    base = p_shot * p_goal
    for _ in range(n_iter):
        new = base + p_move * (T @ xt)
        done = np.abs(new - xt).max() < tol
        xt = new
        if done:
            break
    return xt.reshape(ny, nx)


def xt_lookup(grid, x, y):
    """Return the xT value of the cell containing each (x, y)."""
    ny, nx = grid.shape
    cy, cx = _cell(x, y, nx, ny)
    return grid[cy, cx]


def add_xt(actions, grid):
    """Add xt_start, xt_end and xt_gain columns.

    xt_gain is only defined for moves: a failed move ends with xT 0, so it
    scores a loss equal to the value of where it started.
    """
    out = actions.copy()
    out["xt_start"] = xt_lookup(grid, out["x0"], out["y0"])
    end_val = xt_lookup(grid, out["x1"], out["y1"])
    out["xt_end"] = np.where(out["success"], end_val, 0.0)
    out["xt_gain"] = np.where(out["kind"] == "move", out["xt_end"] - out["xt_start"], np.nan)
    return out