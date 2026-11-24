# pitchxt

**Who controls the dangerous space on a football pitch, and does it predict shots?**

`pitchxt` combines two public ideas in football analytics, then tests the combination against simpler baselines on real tracking data:

- **Expected threat (xT)** says how valuable it is for a team to have the ball in each part of the pitch.
- **Pitch control** (Spearman, 2018) says how likely each team is to win the ball in each part of the pitch, given where every player is and how fast they are moving.

Multiplying the two gives the **value of controlled space**: the dangerous areas a team actually holds at a given moment.

![Value of controlled space](docs/value_of_space.png)

*One frame from Metrica sample game 1. The away team (right panel) controls the space in front of the home goal, which is where threat is highest. Threat share for this frame is 0.23 in favour of the away side.*

## Result

Does a single frame of tracking data say anything about whether a team is about to shoot? For every second of nine matches (two Metrica sample games and seven Bundesliga matches from the DFL open dataset, 219 shots in total), and for each team, I asked whether that team took a shot in the next 5, 10 or 15 seconds. Three indicators are scored by AUC (0.5 is chance, 1.0 is perfect), with 95% intervals from a block bootstrap over 60-second windows.

![Validation](docs/validation_auc.png)

| Score | Shot within 5 s | Shot within 10 s | Shot within 15 s |
|---|---|---|---|
| xT at the ball's location | 0.875 (0.862–0.888) | 0.791 (0.772–0.810) | 0.727 (0.704–0.750) |
| Threat share, Voronoi control × xT | 0.875 (0.856–0.893) | 0.805 (0.778–0.830) | 0.750 (0.718–0.782) |
| Threat share, Spearman control × xT | **0.887** (0.870–0.903) | **0.821** (0.794–0.845) | **0.764** (0.732–0.795) |

Paired comparison at 10 s (difference in AUC on the same resampled windows):

| Comparison | Difference | 95% interval |
|---|---|---|
| Spearman vs xT at the ball | +0.030 | +0.014 to +0.044 |
| Spearman vs Voronoi | +0.015 | +0.012 to +0.019 |
| Voronoi vs xT at the ball | +0.014 | −0.003 to +0.030 |

The two data sources agree on the ranking. At 10 s, Metrica (48 shots) gives 0.821 / 0.838 / 0.853 and DFL (171 shots) gives 0.783 / 0.796 / 0.812 for ball location / Voronoi / Spearman, and the Spearman model is best in both sources at every horizon.

**How to read this honestly:**

- Most of the predictive signal comes from where the ball is: an AUC of about 0.8 at 10 s. Knowing where all 22 players are and how they are moving adds a small but clear improvement, about +0.03 AUC, and the interval now excludes zero.
- The full Spearman model beats the simple nearest-player (Voronoi) version consistently. Voronoi's advantage over the ball's location is **not** established: it disappears at 5 s on the DFL data, and its interval touches zero.
- This is predictive association at the frame level, not a causal claim. Absolute AUCs should not be compared between the two sources: the DFL data marks when the ball is dead and those frames are excluded, which Metrica cannot do, and the original two-game sample was small. The rankings are what carry over.

![Threat share over game 1](docs/threat_share_game1.png)

*Home threat share through Metrica game 1. Red triangles are home shots, blue are away shots; shots tend to follow spells when the shooting team holds the dangerous space.*

## How it works

1. **xT** (`xt.py`): a Markov-chain model after Karun Singh, fitted on ~54,000 open-play passes, carries and shots from the 2022 World Cup (StatsBomb open data) on a 16×12 grid. Cell goal rates are shrunk toward the overall rate so cells with few shots stay stable. Penalties and set-piece restarts are excluded. None of the validation matches are in the xT training data.
2. **Pitch control** (`pitch_control.py`): Spearman's time-to-intercept model, vectorised in NumPy. Each player's arrival time depends on their position, velocity, a 0.7 s reaction time and a 5 m/s top speed, and control is integrated over time until the probabilities sum to one. Parameters are the published defaults and are not tuned. A full 50×32 surface takes about 0.04 s.
3. **Value of space** (`value.py`): control × xT per cell, measured in each team's own attacking direction. *Threat share* is the home team's value divided by the sum of both teams' values.
4. **Data sources** (`tracking.py`, `dfl.py`, `matches.py`): Metrica sample games 1 and 2 and the seven DFL matches (optical tracking, 25 fps, loaded through kloppy) are converted to one common table: metres, centre-origin, home attacking toward +x. The attacking direction is inferred per match from the goalkeeper's position.
5. **Validation** (`validate.py`, `events.py`): frames are sampled once per second while the ball is in play; for each team, the label is whether it shoots within the next N seconds. Confidence intervals come from resampling 60-second blocks, because neighbouring frames are not independent.

## Status and next steps

Done: xT model, pitch control, validation against two baselines on nine matches, interactive explorer.

Next:

1. Fit the Spearman parameters to the data, choosing them on some matches and testing on held-out ones, instead of using the published defaults.
2. Try a graph neural network on the same frames, as a learned alternative to the hand-built threat share.
3. Scale up with more tracking data (for example the PFF 2022 World Cup dataset).

## Limitations

- Nine matches and 219 shots: enough to separate the Spearman model from the baselines, not enough to say precisely how large the improvement is.
- xT is fitted on men's World Cup matches and applied to Bundesliga and Metrica games, so absolute values may not transfer.
- Pitch dimensions for the Metrica data are assumed to be 106 × 68 m. The DFL loader uses the true 105 × 68 m, and the pitch-control grid still uses the 106 × 68 convention, a sub-metre edge effect.
- The Spearman parameters are defaults, not fitted to this data.
- Dead-ball frames are excluded for the DFL matches only, because Metrica has no dead-ball flag.
- Metrica sample game 3 uses a different file format and is not supported yet.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

git clone --depth 1 https://github.com/metrica-sports/sample-data.git data/raw/metrica

pytest -q
python scripts/make_figures.py                # regenerates everything in docs/
python -m streamlit run app/streamlit_app.py  # interactive explorer
python scripts/validate_all.py                # validation on every available match
```

### Optional: DFL data

The DFL files are large and not included in the repo. For each match, download its three XML files into `data/raw/dfl/`. For example, for match `J03WPY`:

```bash
mkdir -p data/raw/dfl && cd data/raw/dfl
BASE=https://huggingface.co/datasets/pysport/idsse-data/resolve/main
curl -L -C - -O $BASE/DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml
curl -L -C - -O $BASE/DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WPY.xml
curl -L -C - -O $BASE/DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml
cd ../../..

python scripts/check_dfl.py J03WPY   # sanity checks and a pitch-control picture
```

The seven match ids are `J03WMX`, `J03WN1`, `J03WPY`, `J03WOH`, `J03WQQ`, `J03WOY` and `J03WR9` (the first two use competition code `DFL-COM-000001`, the rest `DFL-COM-000002`). Anything not downloaded is simply skipped.

## Layout

```
src/pitchxt/
  pitch.py          pitch drawing (StatsBomb units and metres)
  data.py           StatsBomb loader with caching
  xt.py             expected threat model
  tracking.py       Metrica tracking loader, velocities, direction handling
  pitch_control.py  Spearman model and Voronoi baseline
  value.py          control × xT and threat share
  events.py         Metrica event loader
  dfl.py            DFL/Sportec open-data loader (via kloppy)
  matches.py        one entry point for loading any match
  validate.py       shot-prediction validation and bootstrap tests
app/streamlit_app.py    interactive explorer
scripts/
  make_figures.py       regenerates the figures in docs/
  validate_all.py       validation on every available match
  check_dfl.py          sanity checks for the DFL loader
tests/
```

## References and data

- Spearman, W. (2018). *Beyond Expected Goals*. MIT Sloan Sports Analytics Conference.
- Fernández, J. and Bornn, L. (2018). *Wide Open Spaces*. MIT Sloan Sports Analytics Conference.
- Singh, K. (2019). *Introducing Expected Threat (xT)*.
- Laurie Shaw's *Friends of Tracking* tutorials were a helpful reference for working with Metrica data.
- Event data: [StatsBomb open data](https://github.com/statsbomb/open-data). Tracking and event data: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data).
- Tracking and event data from the Deutsche Fußball Liga (DFL), released under CC BY 4.0: Bassek, M., Rein, R., Weber, H. and Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer*. Scientific Data, 12(1), 195. https://doi.org/10.1038/s41597-025-04505-y. Loaded with [kloppy](https://github.com/PySport/kloppy).