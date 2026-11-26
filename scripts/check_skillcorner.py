"""Sanity-check the SkillCorner loader on one match: python scripts/check_skillcorner.py [MATCH_ID]"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pitchxt.pitch_control import plot_pitch_control
from pitchxt.skillcorner import FPS, downloaded_skillcorner, load_skillcorner_match
from pitchxt.tracking import add_velocities, home_attacks_right_first_half, player_ids, to_single_direction

ids = downloaded_skillcorner()
if not ids:
    raise SystemExit("No SkillCorner tracking found (git-lfs pointers only?). See the setup note in scripts/external_test.py")
match_id = sys.argv[1] if len(sys.argv) > 1 else ids[0]
tracking, shots = load_skillcorner_match(match_id)
pids = player_ids(tracking)

print(f"match {match_id}: {len(tracking)} frames at {FPS} fps, periods {sorted(tracking.period.unique())}, "
      f"ball in play {tracking.ball_alive.mean():.0%} of the time")
print("tracking player ids matched the match file via:", tracking.attrs["id_key"], "| pitch:", tracking.attrs["pitch"])
print("players with data:", sum(p.startswith('home') for p in pids), "home,", sum(p.startswith('away') for p in pids), "away")
on = tracking[[f"{p}_x" for p in pids]].notna().sum(axis=1)
print(f"players with a position per frame (min / median / max): {int(on.min())} / {int(on.median())} / {int(on.max())}")
nd = tracking["n_detected"]
print(f"players actually detected on screen per frame: median {nd.median():.0f}; at least 16 in {(nd >= 16).mean():.0%} of frames")
print(f"ball position missing in {tracking.ball_x.isna().mean():.1%} of frames")

print("\n-- direction --")
print("home_team_side from the match file (period 1, period 2):", tracking.attrs["home_team_side"])
print("home attacks right in period 1 (inferred from the keeper):", home_attacks_right_first_half(tracking))
df = to_single_direction(add_velocities(tracking))
for per in (1, 2):
    d = df[df.period == per]
    hx = d[[f"{p}_x" for p in pids if p.startswith("home")]].mean()
    print(f"  after normalising, period {per}: home keeper is {hx.abs().idxmax()} at mean x = {hx.loc[hx.abs().idxmax()]:.1f} (want about -40)")

print("\n-- speeds --")
sp = np.concatenate([np.hypot(df[f"{p}_vx"], df[f"{p}_vy"]).dropna().to_numpy() for p in pids])
print(f"  m/s: median {np.median(sp):.2f}, 99.9th percentile {np.percentile(sp, 99.9):.2f}, max {sp.max():.2f}")

print("\n-- shots --")
print("  per team:", shots.groupby("team").size().to_dict(), "| led to a goal:", int(shots.goal.sum()))
offsets = np.arange(-30, 31, 3)
med = []
for o in offsets:
    f = shots["start_frame"] + o
    bx, by = tracking["ball_x"].reindex(f).to_numpy(), tracking["ball_y"].reindex(f).to_numpy()
    med.append(np.nanmedian(np.hypot(bx - shots["ev_x"].to_numpy(), by - shots["ev_y"].to_numpy())))
at0, best = med[list(offsets).index(0)], int(offsets[int(np.nanargmin(med))])
print(f"  median distance between the shot location and the ball: {at0:.1f} m at offset 0; "
      f"smallest ({min(med):.1f} m) at {best:+d} frames ({best / FPS:+.1f} s)")

play = df[(df.period == 1) & df.ball_alive & df.ball_x.notna()]
f0 = int(play.index[play.index >= play.index.min() + FPS * 900][0])
fig, ax = plt.subplots(figsize=(10, 6.5))
plot_pitch_control(df, f0, ax=ax)
out = "data/processed/skillcorner_check.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nsaved {out} (pitch control at frame {f0})")