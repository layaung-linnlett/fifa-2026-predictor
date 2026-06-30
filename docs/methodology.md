# Methodology

A plain-English explanation of every modelling choice in this project, written so the author can explain it confidently in a job interview without memorising code.

---

## 1. Why Elo ratings?

Elo is a well-established system for ranking competitors based on head-to-head results. It was invented for chess but is widely used in sport, including by FIFA themselves (though they switched to a modified version called the SUM ranking in 2018).

The appeal for this project is that Elo:
- Converts raw match results (win/draw/loss) into a single numerical strength estimate per team
- Naturally handles the fact that beating a strong opponent is more informative than beating a weak one
- Updates incrementally — each new match refines the estimate, which makes it straightforward to compute from a CSV of historical results

The alternative would have been to use FIFA's official world rankings, but Elo computed from scratch from the same data is more transparent, reproducible, and understood.

### Why start from 2014?

The historical dataset contains ~49,000 matches going back to 1872. Using all of them sounds thorough, but creates two problems:

1. **Relevance decay.** A match from 1950 involves players, tactics, and national team compositions that have nothing to do with the 2026 squad. Older results are noise, not signal.
2. **Rating inflation.** Very old teams that no longer exist (e.g. Yugoslavia) accumulate history that can distort the starting ratings of their successor nations.

Starting from January 2014 (two full World Cup cycles before 2026) balances historical depth against relevance. The `--start-date` CLI flag lets users experiment with this choice.

### Match importance weights (K-factors)

The standard Elo system uses a single K-factor for every match. This project uses different K-factors per tournament:

| Tournament | K | Reasoning |
|---|---|---|
| FIFA World Cup | 60 | Highest stakes; teams always play full strength |
| Major continental tournaments | 45 | Serious competition, regional scope |
| Qualifiers | 35 | Competitive but includes weaker opponents |
| Friendlies | 20 | Teams rest players and experiment |
| Other | 30 | Conservative default |

The specific values follow the philosophy of the World Football Elo Ratings (eloratings.net), the most widely cited independent Elo system for international football.

**Key implementation detail:** the K-factor lookup matches tournament names longest-first to prevent "FIFA World Cup" matching inside "FIFA World Cup qualification" and returning the wrong value.

### Form adjustment

Elo is slow to react to short-term changes. A team that just won eight of its last ten matches may still have a lower Elo than a historically strong team that has recently been inconsistent.

To address this, a form adjustment is applied after computing base Elo:

```
adjusted_elo = base_elo + (form_score - 0.5) * 100
```

Where `form_score` is the win rate over the last 10 completed matches (0 = all losses, 1 = all wins, 0.5 = average). With weight = 100, the maximum nudge is ±50 points — a modest adjustment that reacts to hot/cold streaks without overwhelming the base rating.

---

## 2. From Elo to expected goals

To predict scorelines, we need expected goals (xG) for each team — a number representing how many goals they would score on average against this specific opponent.

The conversion used is:

```
p = win_probability(elo_a, elo_b)       # standard Elo formula
xg_a = base_goals * 2 * p
xg_b = base_goals * 2 * (1 - p)
```

With `base_goals = 1.3` (the midpoint of typical international tournament goal rates, roughly 1.2–1.4 goals per team per game), two equally-rated teams both get xG = 1.3. A team with a 90% win probability gets xG ≈ 2.34, their opponent ≈ 0.26.

This is a linear mapping. A more rigorous approach would fit xG directly from historical goal counts using regression, but the linear approximation is consistent and produces sensible values.

---

## 3. Poisson scoring model

Given xG for each team, the Poisson distribution gives the probability of scoring exactly 0, 1, 2, 3... goals:

```
P(team scores k goals) = e^(-xg) * xg^k / k!
```

Because goals are approximately independent events (each goal doesn't directly prevent the other team from scoring), we treat the two teams' goal totals as independent. The probability of a specific scoreline is:

```
P(score i-j) = P(team A scores i) * P(team B scores j)
```

This gives a full 9×9 grid of scoreline probabilities (we cap at 8 goals per team, where the tail probability is negligible).

### Dixon-Coles correction

Plain Poisson has a known flaw in football: it systematically mis-predicts four low-scoring results:

| Scoreline | Plain Poisson | Reality |
|---|---|---|
| 0-0 | Over-predicts | Happens less often |
| 1-0 | Under-predicts | Happens more often |
| 0-1 | Under-predicts | Happens more often |
| 1-1 | Over-predicts | Happens less often |

This is because in real matches, teams adjust their behaviour based on the current scoreline. When a match is goalless, both teams tend to attack more, making a 1-0 result more likely than independent Poisson predicts. When a team is winning 1-0, they become more conservative, making 1-1 less likely.

Dixon and Coles (1997) derived a correction factor, `tau`, that multiplies the raw Poisson probability for these four scorelines:

```
tau(0,0) = 1 - xg_a * xg_b * rho        # reduces 0-0 probability
tau(1,0) = 1 + xg_b * rho               # increases 1-0 probability
tau(0,1) = 1 + xg_a * rho               # increases 0-1 probability
tau(1,1) = 1 - rho                       # reduces 1-1 probability
tau(i,j) = 1   for all other scorelines  # no change
```

After applying the correction, all probabilities are renormalised to sum to 1.

**rho = 0.1** was chosen conservatively. Dixon and Coles estimated rho ≈ 0.13 from English club football data. Using 0.1 applies a slightly weaker correction, appropriate given that international football data (which includes more varied playing styles and tactical contexts) may not perfectly match the original paper's calibration.

---

## 4. EV-optimal prediction

The competition awards:
- 25 points for the exact scoreline
- 10 points for the correct result (home win / draw / away win) regardless of goals
- 0 points for the wrong result

For each possible prediction (i, j), the expected points are:

```
EV(i,j) = P(exact score i-j) * 25 + P(correct result, wrong score) * 10
         = P(i,j) * 25 + (P(same result) - P(i,j)) * 10
```

The scoreline that maximises this expected value is chosen as the prediction.

**Why this differs from the most likely scoreline:** suppose the most likely scoreline is 1-1 with probability 18%, and the home team wins with probability 60%. Predicting 1-1 earns EV = 0.18 × 25 + 0.18 × 10 = 4.5 + 1.8 = 6.3 points (since 1-1 is a draw, not a home win). Predicting 1-0 (a home win result) earns EV = P(1-0) × 25 + 0.60 × 10 = say 4.75 + 6.0 = 10.75 points, despite 1-0 being less likely than 1-1 as an exact score.

This is the key insight: in competitions with partial credit, the best prediction is not always the most probable outcome.

---

## 5. Corners and cards

The competition requires corners and cards predictions for every match.

**Ideal approach:** load a historical dataset of match statistics (corners, yellow cards, red cards per game), compute each team's average per game, and model predictions as the sum of two teams' averages adjusted for opponent strength.

**Actual approach:** the available dataset (`results.csv`) only records goals, not match statistics. A corners/cards dataset for international football is not freely available in the same clean format.

Instead, this model uses:
- **Base rates** from World Cup tournament averages: ~10 corners, ~3 yellow cards, ~0.15 red cards per match
- **Style multipliers** per team: qualitative estimates of whether a team tends to win more/fewer corners than average, and whether they tend to receive more/fewer cards

These multipliers are clearly labelled as estimates throughout the codebase. They are informed by general knowledge of playing styles (Spain's possession football wins many corners; Japan's disciplined pressing earns few cards) but are not statistically fitted from data.

**This is documented as a known limitation**, not concealed. Transparency about what is modelled vs estimated is a core data science practice.

---

## 6. Tournament simulation

### Group stage

All 72 group matches are simulated by sampling a random scoreline from the Poisson probability table for each match. This is equivalent to spinning a weighted roulette wheel where each pocket represents a scoreline, sized by its probability.

Group standings follow FIFA tiebreaker rules: points → goal difference → goals scored → head-to-head points → head-to-head goal difference.

The 8 best third-place teams are selected by ranking all 12 third-place finishers by the same criteria and taking the top 8.

### Knockout stage

Knockout matches are resolved the same way. If the sampled score is a draw, a penalty shootout is simulated: the higher-Elo team wins with probability 0.5 + a small edge (capped at ±10%), reflecting that shootouts are largely 50/50 but better teams have a slight advantage.

### Bracket seeding

The 2026 bracket is fixed — FIFA pre-determines which group slots go to which bracket positions. The slot descriptions (`"Winner Group C"`, `"Runner-up Group F"`) are resolved to actual team names after the group stage is simulated.

For the "Best 3rd" slots (which list 5 candidate groups each), teams are assigned in rank order to the first available slot that lists their group. The official FIFA placement table for this format was not published at time of writing; this approximation may differ slightly from the official rules.

### Monte Carlo aggregation

The full process (group stage + knockout) is repeated 10,000 times. The fraction of runs in which each team wins the tournament is their title probability. At 10,000 simulations, probabilities are stable to approximately ±0.5%.

The random seed is fixed (default: 42) for reproducibility. Changing `--seed 0` produces a fresh random result each run.

---

## 7. Competition scoring

Points follow a two-factor system:

| Accuracy | Base points |
|---|---|
| Exact scoreline | 25 |
| Correct result only | 10 |
| Wrong result | 0 |

Multiplied by the round multiplier (1× group stage through to 16× Final), giving a maximum of 400 points for a correct exact score in the Final.

The `scoring.py` module implements this and can evaluate any set of predictions against actual results — useful for backtesting the model on historical tournaments.

---

## 8. What this model cannot do

- **Squad selection and injuries:** Elo treats every match as if both teams play full strength.
- **Tactical matchups:** some teams with lower Elo consistently trouble specific opponents due to tactical style. This is not captured.
- **Weather, pitch, travel fatigue:** not modelled.
- **Market information:** betting odds aggregate huge amounts of information (injuries, lineups, recent news). This model does not use that information.
- **Very short odds:** the model will not correctly price a 1-in-1000 upset; the Poisson tail at extreme scores is noisy.

These limitations are not reasons to distrust the model — they are reasons to understand it correctly. For a portfolio project demonstrating statistical thinking, model engineering, and honest communication of uncertainty, it is more than sufficient.
