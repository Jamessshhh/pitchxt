"""Pitch drawing in StatsBomb coordinates (120 x 80, origin top-left)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

L, W = 120, 80


def draw_pitch(ax=None, line_color="#444", bg="white"):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor(bg)
    kw = dict(fill=False, ec=line_color, lw=1.5)

    # outline, halfway line, centre circle
    ax.add_patch(Rectangle((0, 0), L, W, **kw))
    ax.plot([L / 2, L / 2], [0, W], color=line_color, lw=1.5)
    ax.add_patch(Circle((L / 2, W / 2), 9.15, **kw))

    # penalty areas (18 yd x 44 yd) and six-yard boxes (6 yd x 20 yd)
    for x0 in (0, L - 18):
        ax.add_patch(Rectangle((x0, W / 2 - 20.16), 18, 40.32, **kw))
    for x0 in (0, L - 6):
        ax.add_patch(Rectangle((x0, W / 2 - 9.16), 6, 18.32, **kw))

    ax.set_xlim(-2, L + 2)
    ax.set_ylim(W + 2, -2)  # y increases downward in StatsBomb data
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_grid(grid, ax=None, cmap="YlOrRd", title=None):
    """Overlay a (n_y, n_x) grid, such as an xT map, on the full pitch."""
    ax = draw_pitch(ax)
    im = ax.imshow(grid, extent=(0, L, W, 0), cmap=cmap, alpha=0.8, zorder=0)
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.7)
    return ax