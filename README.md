# FIFA World Cup 2026 Prediction Engine

A clean, reproducible Python package that predicts outcomes for the 2026 FIFA World Cup — 72 group-stage matches, 32 knockout matches, and full tournament odds — using Elo ratings, a Dixon-Coles corrected Poisson model, and Monte Carlo simulation.

Built as a portfolio project for graduate data analyst / data scientist roles.

---

## Methodology (plain English)

**Step 1 — Elo ratings.** Every international team is given a numerical strength rating, updated after every match since 2014. Bigger wins move the rating more. World Cup matches count three times more than friendlies, because they are a more reliable signal of true team strength.

**Step 2 — Expected goals.** Before each match, the Elo rating gap is converted into expected goals (xG) for each team — a number representing how many goals they would score on average against this opponent.

**Step 3 — Dixon-Coles Poisson model.** The xG figures are fed into a Poisson distribution to produce a full probability table over all possible scorelines (0-0, 1-0, 2-1, ...). A small correction (Dixon & Coles, 1997) adjusts the probabilities of very low-scoring results, which plain Poisson systematically mis-predicts.

**Step 4 — EV-optimal prediction.** Rather than always predicting the single most likely scoreline, we compute the *expected points* for every possible prediction under the competition scoring rules, and pick the scoreline with the highest expected value.

**Step 5 — Monte Carlo simulation.** The full tournament is simulated 10,000 times. Each run samples random scorelines from the Poisson model, so upsets happen at the right frequency. The fraction of runs each team wins is their title probability.

---

## Example output — title odds (10,000 simulations)

| Team | Title % | Finalist % | Semi-final % |
|------|--------:|----------:|-------------:|
| Argentina | ~18% | ~28% | ~42% |
| Spain | ~15% | ~26% | ~44% |
| France | ~12% | ~21% | ~36% |
| Germany | ~9% | ~17% | ~28% |
| Morocco | ~6% | ~10% | ~20% |
| Japan | ~5% | ~10% | ~21% |
| England | ~4% | ~9% | ~18% |

*Results vary slightly by random seed. Run `simulate` to generate fresh numbers.*

---

## Project structure

```
fifa-2026-predictor/
├── src/fifa_predictor/
│   ├── elo.py              # Elo ratings with match importance weights
│   ├── poisson_model.py    # Dixon-Coles corrected Poisson model
│   ├── cards_corners.py    # Corners / cards estimation
│   ├── simulation.py       # Monte Carlo group + knockout simulation
│   ├── bracket.py          # Fixed FIFA 2026 bracket seeding logic
│   ├── scoring.py          # Competition points scoring
│   └── cli.py              # Command-line interface (click)
├── data/raw/
│   ├── results.csv         # ~49,000 historical international results
│   ├── group_fixtures.csv  # 72 group-stage fixtures
│   └── knockout_slots.csv  # 32 knockout bracket slots + multipliers
├── tests/                  # pytest unit + integration tests (77 tests)
├── outputs/predictions/    # Generated CSVs (gitignored)
└── docs/methodology.md     # Full technical writeup
```

---

## Setup

```bash
# Clone and install
git clone <your-repo-url>
cd fifa-2026-predictor
pip install -e .
```

Requires Python 3.10+. Dependencies: `pandas`, `scipy`, `numpy`, `click`.

---

## How to run

### Predict all 72 group-stage matches
```bash
python -m fifa_predictor predict-groups
# Output: outputs/predictions/group_predictions.csv
```

Each row includes: predicted scoreline, win/draw/loss probabilities, expected competition points, corners and cards estimates.

### Run Monte Carlo simulation (title odds)
```bash
python -m fifa_predictor simulate --sims 10000
# Output: outputs/predictions/title_odds.csv + printed leaderboard
```

Runs the full tournament 10,000 times and aggregates title, finalist, and semi-final probabilities per team. Takes a few minutes.

### Predict the full knockout bracket
```bash
python -m fifa_predictor predict-knockout
# Output: outputs/predictions/knockout_predictions.csv + printed bracket
```

Simulates the group stage to determine the most likely qualifiers, then predicts each knockout match in bracket order.

### Options
```bash
python -m fifa_predictor simulate --sims 5000 --seed 0   # different result each run
python -m fifa_predictor predict-groups --start-date 2018-01-01
python -m fifa_predictor --help
```

---

## Running the tests

```bash
python -m pytest tests/ -v
```

77 tests covering Elo update maths, Poisson probability correctness (sum to 1), Dixon-Coles correction bounds, scoring logic, and an integration test for the simulation pipeline.

---

## Limitations (honest section)

**Corners and cards are estimated, not data-driven.** The historical dataset only records goals. Per-team style multipliers for corners and cards are informed qualitative estimates. They should be treated as approximate, not as statistically fitted values.

**Elo does not know about injuries or lineups.** A key player being injured or suspended before a match could shift the real probability significantly. The model treats each match as if both teams field full-strength squads.

**Playoff teams are unknowns.** Several 2026 qualifier spots were still undecided at time of writing. Playoff slots are given neutral Elo ratings of 1500.

**Dixon-Coles rho is not fitted from data.** The correction parameter (rho = 0.1) was set conservatively rather than estimated from historical international match data. A properly fitted value would require a dedicated parameter estimation step (maximum likelihood estimation on the historical dataset).

**10,000 simulations is enough for stable percentages but not exact.** Title probabilities are stable to roughly ±0.5% at 10,000 runs. Doubling to 20,000 halves that variance but takes twice as long.

**No home advantage.** For a tournament played at neutral venues in the USA, Canada, and Mexico, no home advantage adjustment is applied. USA and Mexico players could plausibly benefit slightly.

---

## Interview talking points

1. **"I implemented a Dixon-Coles correction on top of the Poisson model."** Standard Poisson over-predicts 0-0 draws and under-predicts 1-0 results. Dixon and Coles (1997) derived a small multiplicative fix for these four low-score results. I implemented it and documented exactly what each term does.

2. **"I use EV-optimal predictions rather than the most likely score."** In a competition with partial credit for getting the result right, the most likely scoreline is not always the best prediction. I compute the expected points for every possible scoreline and pick the one that maximises them. This is a decision theory idea, not just "predict the obvious".

3. **"The model is backtestable."** The historical results file goes back to 1872. I can hold out any period, re-run the Elo computation, and measure how well the model would have predicted those real outcomes. This makes the model auditable rather than just plausible-looking.

4. **"I was honest about what the model does and doesn't know."** The limitations section explicitly flags estimated vs data-driven quantities. A model that overclaims precision is worse than one that is clear about its assumptions.
