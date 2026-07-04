# FIFA World Cup 2026 Prediction Engine

Predicts every match of the 2026 FIFA World Cup using historical data, statistical modelling, and 10,000 simulated tournaments.

---

## What this project does

It takes 49,000 historical international football results and uses them to answer one question: **who is most likely to win the 2026 World Cup, and by how much?**

For every match — all 104 of them — the model predicts:
- The most strategically optimal scoreline to submit
- The probability of each possible result (home win / draw / away win)
- Estimated corners and cards

For the full tournament, it outputs the probability of each team reaching the semi-finals, final, and winning the title.

---

## How the model works

**Step 1 — Team strength (Elo ratings)**
Every team gets a numerical strength score, updated after every international match since 2014. Beating a strong team moves your score up a lot. Losing to a weak team drops it significantly. World Cup matches count three times more than friendlies, because they are a better indicator of true team quality.

**Step 2 — Match forecast (Expected goals)**
The strength gap between two teams is converted into expected goals — how many goals each team would typically score against that opponent.

**Step 3 — Scoreline probabilities (Dixon-Coles Poisson model)**
Expected goals feed into a statistical model that calculates the probability of every possible scoreline: 0-0, 1-0, 2-1, and so on. A correction from Dixon & Coles (1997) fixes known errors that the standard model makes on low-scoring results.

**Step 4 — Best prediction (Expected value)**
Rather than simply predicting the most likely score, the model picks the scoreline that maximises expected competition points. These are often different — for example, predicting a 1-0 home win can earn more expected points than predicting a 1-1 draw, even if 1-1 is slightly more probable.

**Step 5 — Tournament simulation (Monte Carlo)**
The entire tournament — all 72 group matches and 32 knockout matches — is simulated 10,000 times. Each run samples random results weighted by their probabilities, so upsets happen naturally. The percentage of runs a team wins is their title probability.

---

## Results — predicted title odds

| Team | Win the title | Reach the final | Reach the semi-final |
|---|---:|---:|---:|
| Argentina | ~23% | ~34% | ~49% |
| Spain | ~16% | ~27% | ~42% |
| France | ~12% | ~21% | ~35% |
| Germany | ~8% | ~16% | ~30% |
| Morocco | ~7% | ~13% | ~23% |
| England | ~4% | ~9% | ~19% |
| Portugal | ~4% | ~10% | ~21% |

**Predicted winner: Argentina**, who win the tournament in roughly 1 in 4 simulations — reflecting their current form and Elo rating, not certainty. Any of the top teams can win on the day.

*(Table generated from `fifa-predictor simulate --sims 10000 --seed 42`.)*

---

## Project structure

```
fifa-2026-predictor/
├── src/fifa_predictor/
│   ├── elo.py              # Team strength ratings from historical results
│   ├── poisson_model.py    # Scoreline probability model
│   ├── cards_corners.py    # Corners and cards estimates
│   ├── simulation.py       # Monte Carlo tournament simulator
│   ├── bracket.py          # Knockout bracket seeding logic
│   ├── scoring.py          # Competition points calculator
│   └── cli.py              # Command-line interface
├── data/raw/
│   ├── results.csv         # 49,477 historical international results, 1872-2026
│   ├── group_fixtures.csv  # All 72 group-stage fixtures
│   └── knockout_slots.csv  # Knockout bracket structure
├── tests/                  # 77 automated tests
└── docs/methodology.md     # Full technical explanation
```

**Data source:** `results.csv` is the [International football results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) dataset (Mart Jürisoo, Kaggle), 3.6 MB, included in this repo as-is.

---

## Getting started

**Requirements:** Python 3.10+

```bash
git clone https://github.com/layaung-linnlett/fifa-2026-predictor.git
cd fifa-2026-predictor
pip install -e .
```

---

## Running the model

**Predict all 72 group-stage matches:**
```bash
python -m fifa_predictor predict-groups
```
Saves a CSV with predicted scores, win probabilities, and corners/cards for every match.

**Run the full tournament simulation:**
```bash
python -m fifa_predictor simulate --sims 10000
```
Simulates the entire World Cup 10,000 times and prints the title odds leaderboard.

**Predict the knockout bracket:**
```bash
python -m fifa_predictor predict-knockout
```
Finds the most likely bracket path and predicts every knockout match from Round of 32 to Final.

---

## Running the tests

```bash
python -m pytest tests/ -v
```

62 tests covering every module — from the Elo update formula to the full simulation pipeline.

---

## Limitations

- **Corners and cards are estimates.** The dataset only records goals, so these figures are based on each team's known playing style rather than measured statistics.
- **No injury or squad information.** The model assumes both teams play full strength. A key injury could significantly change the real probability.
- **Playoff teams are unknown.** Teams yet to qualify through playoffs are given an average rating.
- **No home advantage.** The tournament is hosted across the USA, Canada, and Mexico. No adjustment is made for local support.

---

## Further reading
- [docs/methodology.md](docs/methodology.md) — detailed walkthrough of every modelling decision
