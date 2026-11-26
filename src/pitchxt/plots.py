"""Chart helpers shared by the figure script."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BAR_COLOURS = ["#9aa5b1", "#d9a441", "#5b8def", "#d6455d", "#7b2d8e"]
LINE_STYLES = {"xT at ball": ("#9aa5b1", "o"), "depth": ("#d9a441", "s"),
               "published": ("#d6455d", "^"), "look-ahead": ("#7b2d8e", "D")}


def validation_chart(results, horizons, title, path):
    """Grouped bars of AUC with 95% intervals: one group per horizon, one bar per score."""
    names = list(results[0]["score"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        aucs = np.array([r.loc[r.score == name, "auc"].iloc[0] for r in results])
        lo = np.array([r.loc[r.score == name, "ci_low"].iloc[0] for r in results])
        hi = np.array([r.loc[r.score == name, "ci_high"].iloc[0] for r in results])
        ax.bar(np.arange(len(horizons)) + (i - (len(names) - 1) / 2) * width, aucs, width, label=name,
               color=BAR_COLOURS[i], yerr=[aucs - lo, hi - aucs], capsize=3, ecolor="#333")
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"shot within {h} s" for h in horizons])
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="#999", lw=1, ls="--")
    ax.set_ylabel("AUC (0.5 = chance)")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def detection_chart(bins, title, path):
    """AUC of the main scores against how many players were actually seen on screen."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, (colour, marker) in LINE_STYLES.items():
        ax.plot(bins["detected"], bins[name], marker=marker, color=colour, lw=2, label=name)
    for i, (frames, pos) in enumerate(zip(bins["frames"], bins["positive"])):
        ax.annotate(f"{frames / 1000:.0f}k frames\n{pos} positive", (i, 0.505), ha="center", fontsize=7, color="#666")
    ax.set_ylim(0.5, 0.95)
    ax.set_xlabel("players actually detected on screen (the rest are extrapolated)")
    ax.set_ylabel("AUC, shot within 10 s")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)