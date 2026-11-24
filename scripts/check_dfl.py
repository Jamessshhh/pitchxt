"""Sanity-check the DFL loader on one match: python scripts/check_dfl.py [MATCH_ID]"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pitchxt.dfl import FPS, load_dfl_match
from pitchxt.pitch_control import plot_pitch_control
from pitchxt.tracking import (add_velocities, home_attacks_right_first_half, player_ids,
                              to_single_direction)

match_id = sys.argv[1] if len(sys.argv) > 1 else "J03WPY"
tracking, shots = load_dfl_match(match_id)
ids = player_ids(tracking)

print(f"match {match_id}: {len(tracking)} frames, periods {sorted(tracking.period.unique())}, "
      f"ball alive {tracking.ball_alive.mean():.1%} of the time")
print("tracked players:", sum(i.startswith('home') for i in ids), "home,",
      sum(i.startswith('away') for i in ids), "away")
for per in (1, 2):
    d = tracking[tracking.period == per]
    n = d[[f"{p}_x" for p in ids]].notna().sum(axis=1)
    print(f"  period {per}: players on the pitch per frame (min / median / max): "
          f"{int(n.min())} / {int(n.median())} / {int(n.max())}")

print("\n-- direction --")
print("home attacks right in period 1 (inferred from the keeper):", home_attacks_right_first_half(tracking))
df = to_single_direction(add_velocities(tracking))
for per in (1, 2):
    d = df[df.period == per]
    hx = d[[f"{p}_x" for p in ids if p.startswith("home")]].mean()
    print(f"  after normalising, period {per}: home keeper is {hx.abs().idxmax()} at mean x = {hx.loc[hx.abs().idxmax()]:.1f} (want about -40)")

print("\n-- speeds --")
sp = np.concatenate([np.hypot(df[f"{p}_vx"], df[f"{p}_vy"]).dropna().to_numpy() for p in ids])
print(f"  m/s: median {np.median(sp):.2f}, 99.9th percentile {np.percentile(sp, 99.9):.2f}, max {sp.max():.2f}")

print("\n-- shots --")
print("  per team:", shots.groupby("team").size().to_dict(), "| goals:", int(shots.goal.sum()))
offsets = np.arange(-75, 76, 5)
med = []
for o in offsets:
    f = shots["start_frame"] + o
    bx = tracking["ball_x"].reindex(f).to_numpy()
    by = tracking["ball_y"].reindex(f).to_numpy()
    med.append(np.nanmedian(np.hypot(bx - shots["ev_x"].to_numpy(), by - shots["ev_y"].to_numpy())))
at0 = med[list(offsets).index(0)]
best = int(offsets[int(np.nanargmin(med))])
print(f"  median distance between the shot location and the ball: {at0:.1f} m at offset 0; "
      f"smallest ({min(med):.1f} m) at {best:+d} frames ({best / FPS:+.1f} s)")

first_half = df[(df.period == 1) & df.ball_alive & df.ball_x.notna()]
f0 = int(first_half.index[first_half.index >= first_half.index.min() + FPS * 900][0])  # ~15 min in, ball in play
fig, ax = plt.subplots(figsize=(10, 6.5))
plot_pitch_control(df, f0, ax=ax)
out = "data/processed/dfl_check.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nsaved {out} (pitch control at frame {f0})")