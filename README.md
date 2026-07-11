# FIFA World Cup 2026 Prediction Engine

Predicts every match of the 2026 FIFA World Cup using historical data, statistical modelling, and 10,000 simulated tournaments.

---

## Key findings

It takes 49,000 historical international football results and uses them to answer one question: **who is most likely to win the 2026 World Cup, and by how much?**

For every match — all 104 of them — the model predicts:
- The most strategically optimal scoreline to submit
- The probability of each possible result (home win / draw / away win)
- Estimated corners and cards

For the full tournament, it outputs the probability of each team reaching the semi-finals, final, and winning the title.

| Team | Win the title | Reach the final | Reach the semi-final |
|---|---:|---:|---:|
| Argentina | ~18% | ~28% | ~42% |
| Spain | ~15% | ~26% | ~44% |
| France | ~12% | ~21% | ~36% |
| Germany | ~9% | ~17% | ~28% |
| Morocco | ~6% | ~10% | ~20% |
| Japan | ~5% | ~10% | ~21% |
| England | ~4% | ~9% | ~18% |

**Predicted winner: Argentina**, who win the tournament in roughly 1 in 5 simulations — reflecting their current form and Elo rating, not certainty. Any of the top teams can win on the day.

---

## Screenshots

No exported chart images yet — the Elo ratings bar chart, the France vs Germany scoreline heatmap, and the title-odds bar chart are all generated live in `notebooks/fifa_prediction.ipynb`. Open the notebook to see them, or export a couple as PNGs into this section.

---

## Tech stack

- **Python 3.10+**
- **pandas / numpy** — data loading and manipulation
- **scipy.stats** — Poisson distribution for scoreline probabilities
- **click** — command-line interface
- **matplotlib** — charts in the exploration notebook
- **pytest** — test suite (77 tests)
- **GitHub Actions** — runs the test suite on every push and pull request to `main`

---

## Methodology

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

See [docs/methodology.md](docs/methodology.md) for a detailed walkthrough of every modelling decision, including the reasoning behind each judgment call (K-factors, the 2014 start date, rho = 0.1, and so on).

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
│   ├── results.csv         # 49,000 historical international results
│   ├── group_fixtures.csv  # All 72 group-stage fixtures
│   └── knockout_slots.csv  # Knockout bracket structure
├── notebooks/
│   └── fifa_prediction.ipynb  # End-to-end walkthrough with charts
├── tests/                  # 77 automated tests
└── docs/methodology.md     # Full technical explanation
```

---

## How to run

**Requirements:** Python 3.10+

```bash
git clone https://github.com/layaung-linnlett/fifa-2026-predictor.git
cd fifa-2026-predictor
pip install -e .
```

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

**Run the tests:**
```bash
pip install -e ".[test]"
python -m pytest tests/ -v
```
77 tests covering every module — from the Elo update formula to the full simulation pipeline.

**Run the notebook:**
```bash
pip install -e ".[notebook]"
jupyter notebook notebooks/fifa_prediction.ipynb
```

---

## Limitations & future work

- **Corners and cards are estimates.** The dataset only records goals, so these figures are based on each team's known playing style rather than measured statistics. Fitting this from real match-stats data would need a corners/cards dataset that isn't freely available in the same clean form as the goals data — worth revisiting if one turns up.
- **No injury or squad information.** The model assumes both teams play full strength. A key injury could significantly change the real probability.
- **Playoff teams are unknown.** Teams yet to qualify through playoffs are given an average rating. Ratings should be re-run once the playoff draws are confirmed.
- **No home advantage.** The tournament is hosted across the USA, Canada, and Mexico. No adjustment is made for local support.

---

## Contact

[github.com/layaung-linnlett](https://github.com/layaung-linnlett)
