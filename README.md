# FIFA World Cup 2026 Prediction

A Python package for predicting the 2026 FIFA World Cup — covering all 72 group-stage matches, 32 knockout matches, and full tournament odds — built on Elo ratings, a Dixon-Coles-corrected Poisson model, and Monte Carlo simulation.

---

## Overview

| | |
|---|---|
| **Matches predicted** | 104 (72 group stage + 32 knockout) |
| **Simulations** | 10,000 Monte Carlo runs |
| **Historical data** | ~49,000 international results (2014–2026) |
| **Test coverage** | 62 unit and integration tests |
| **Stack** | Python 3.10 · pandas · scipy · click · pytest |

---

## How it works

The pipeline has five stages:

**1. Elo ratings** — Each team's strength is represented as a single number, updated after every match since 2014. World Cup results move ratings 3× more than friendlies, reflecting their greater reliability as a signal of true team quality.

**2. Expected goals (xG)** — The Elo gap between two teams is converted into expected goals per team, calibrated to the international tournament average of ~1.3 goals per team per game.

**3. Dixon-Coles Poisson model** — xG figures are fed into a Poisson distribution to produce a full scoreline probability table. A correction from Dixon & Coles (1997) fixes the four low-scoring results that plain Poisson systematically mis-predicts (0-0, 1-0, 0-1, 1-1).

**4. EV-optimal prediction** — Rather than always predicting the most likely scoreline, the model computes expected competition points for every possible prediction and selects the one with the highest expected value. The most probable score and the best prediction are often different.

**5. Monte Carlo simulation** — The full 48-team tournament is simulated 10,000 times, sampling random scorelines from the Poisson model on each run. The fraction of runs a team wins equals their title probability.

---

## Example output — title odds

| Team | Title % | Finalist % | Semi-final % |
|---|---:|---:|---:|
| Argentina | ~18% | ~28% | ~42% |
| Spain | ~15% | ~26% | ~44% |
| France | ~12% | ~21% | ~36% |
| Germany | ~9% | ~17% | ~28% |
| Morocco | ~6% | ~10% | ~20% |
| Japan | ~5% | ~10% | ~21% |
| England | ~4% | ~9% | ~18% |

*Generated with `--sims 10000 --seed 42`. Results vary slightly by random seed.*

---

## Project structure

```
fifa-2026-predictor/
├── src/fifa_predictor/
│   ├── elo.py              # Elo ratings with match importance K-factors
│   ├── poisson_model.py    # Dixon-Coles corrected Poisson model
│   ├── cards_corners.py    # Corners and cards estimation
│   ├── simulation.py       # Monte Carlo group + knockout simulation
│   ├── bracket.py          # Fixed 2026 knockout bracket seeding
│   ├── scoring.py          # Competition points logic with round multipliers
│   └── cli.py              # Command-line interface
├── data/raw/
│   ├── results.csv         # ~49,000 historical international results
│   ├── group_fixtures.csv  # 72 group-stage fixtures
│   └── knockout_slots.csv  # 32 knockout bracket slots + multipliers
├── tests/                  # 62 pytest unit and integration tests
├── docs/methodology.md     # Full technical writeup
└── outputs/predictions/    # Generated CSVs (gitignored)
```

---

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/layaung-linnlett/fifa-2026-predictor.git
cd fifa-2026-predictor
pip install -e .
```

---

## Usage

### Predict group-stage matches

```bash
python -m fifa_predictor predict-groups
# -> outputs/predictions/group_predictions.csv
```

Outputs predicted scoreline, win/draw/loss probabilities, expected competition points, and corners/cards estimates for all 72 matches.

### Run the Monte Carlo simulation

```bash
python -m fifa_predictor simulate --sims 10000
# -> outputs/predictions/title_odds.csv + printed leaderboard
```

Runs the full tournament 10,000 times and prints title, finalist, and semi-final probabilities per team.

### Predict the knockout bracket

```bash
python -m fifa_predictor predict-knockout
# -> outputs/predictions/knockout_predictions.csv + printed bracket
```

Simulates the group stage to identify the most likely qualifiers, then predicts each knockout match in bracket order.

### Additional options

```bash
python -m fifa_predictor simulate --sims 5000 --seed 0      # fresh random result each run
python -m fifa_predictor predict-groups --start-date 2018-01-01
python -m fifa_predictor --help
```

---

## Tests

```bash
python -m pytest tests/ -v
```

62 tests covering Elo update arithmetic, Poisson probability normalisation, Dixon-Coles correction bounds, EV-optimal prediction logic, scoring rules, and an end-to-end simulation integration test.

---

## Limitations

**Corners and cards are not data-driven.** The historical dataset records only goals. Per-team style multipliers for corners and cards are informed estimates, not statistical fits, and should be treated as approximations.

**No injury or lineup information.** The model assumes both teams field full-strength squads. A key absence can shift real probabilities significantly.

**Playoff teams have neutral ratings.** Several 2026 qualification spots were undecided at time of writing. Unresolved playoff slots are assigned a default Elo of 1500.

**Dixon-Coles rho is not fitted from data.**  The correction parameter (rho = 0.1) was set conservatively. A properly estimated value would require maximum likelihood estimation on historical goal data.
---

## Further reading
- [docs/methodology.md](docs/methodology.md) — full walkthrough of every modelling decision in this project
