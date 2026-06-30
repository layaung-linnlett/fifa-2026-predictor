"""
Poisson scoring model with Dixon-Coles low-score correction.

--- Plain English explanation ---

PLAIN POISSON:
    Goals in football arrive roughly randomly throughout a match.
    If we expect England to score 2.0 goals on average, the Poisson
    distribution tells us the probability of them scoring exactly 0, 1, 2,
    3... goals. We do this independently for both teams, then multiply:

        P(England 2 - Germany 1) = P(England score 2) * P(Germany score 1)

    This gives us a full grid of scoreline probabilities.

WHY PLAIN POISSON IS WRONG FOR LOW SCORES:
    Real football data shows that 0-0, 1-0, 0-1, and 1-1 happen at different
    rates than plain Poisson predicts. Specifically:
        - 0-0 draws happen LESS often than Poisson says
        - 1-0 and 0-1 wins happen MORE often than Poisson says
        - 1-1 draws happen LESS often than Poisson says

    This is because in real matches, teams adjust their behaviour when the
    score is low — a team losing 0-1 pushes forward and creates chances,
    which the independent Poisson model doesn't account for.

DIXON-COLES CORRECTION (the tau function):
    Dixon and Coles (1997) published a small multiplicative fix for exactly
    these four scorelines. We multiply the raw Poisson probability by a
    correction factor tau(i, j, xg_a, xg_b, rho):

        corrected_prob = raw_poisson_prob * tau(i, j, xg_a, xg_b, rho)

    tau equals:
        (0,0):  1 - xg_a * xg_b * rho
        (1,0):  1 + xg_b * rho
        (0,1):  1 + xg_a * rho
        (1,1):  1 - rho
        else:   1  (no correction for higher scores)

    rho is a small positive number (we use 0.1) that controls how strong
    the correction is. A higher rho = stronger correction. After applying
    tau, the probabilities no longer sum to exactly 1.0, so we renormalise.

JUDGMENT CALL — rho = 0.1:
    Dixon and Coles estimated rho from real data at around 0.13. We use 0.1
    as a slightly conservative value. The difference in predictions is tiny
    (less than 1% on any given scoreline) but the correction is principled
    and shows you understand the literature, which matters for a portfolio.
"""

from scipy.stats import poisson


# ---------------------------------------------------------------------------
# JUDGMENT CALL: base_goals = 1.3
#
# We convert an Elo win-probability into expected goals (xG) using:
#     xg_a = base_goals * 2 * p_win_a
#     xg_b = base_goals * 2 * (1 - p_win_a)
#
# So if two equal teams play (p = 0.5), both get xG = 1.3.
# International football averages roughly 1.2–1.4 goals per team per game,
# so 1.3 is the midpoint of that empirical range.
# ---------------------------------------------------------------------------
BASE_GOALS = 1.3

# JUDGMENT CALL: rho = 0.1 (see module docstring above)
DIXON_COLES_RHO = 0.1


def win_probability(elo_a: float, elo_b: float) -> float:
    """
    Convert Elo ratings into a win probability for team A.

    P(A wins) = 1 / (1 + 10^((elo_b - elo_a) / 400))

    This is the same formula used in chess Elo. The 400 scaling constant
    means a 400-point gap corresponds to ~91% expected win rate.
    """
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def expected_goals(elo_a: float, elo_b: float, base: float = BASE_GOALS) -> tuple:
    """
    Convert Elo ratings into expected goals (xG) for each team.

    We scale the win probability so that:
        - Equal teams both get xG = base (1.3)
        - A dominant team (p=0.9) gets xG ≈ 2.34, opponent ≈ 0.26

    This is a simple linear mapping. More sophisticated models fit xG
    from actual historical goal data, but this is a sound starting point.
    """
    p = win_probability(elo_a, elo_b)
    xg_a = base * 2 * p
    xg_b = base * 2 * (1 - p)
    return round(xg_a, 4), round(xg_b, 4)


def _dixon_coles_tau(i: int, j: int, xg_a: float, xg_b: float, rho: float) -> float:
    """
    Dixon-Coles correction factor for low-scoring results.

    Only modifies the four scorelines (0,0), (1,0), (0,1), (1,1).
    Returns 1.0 for all other scorelines (no correction needed).

    The intuition:
        rho > 0 makes 0-0 and 1-1 LESS likely (they're over-predicted).
        rho > 0 makes 1-0 and 0-1 MORE likely (they're under-predicted).
    """
    if i == 0 and j == 0:
        return 1 - xg_a * xg_b * rho
    elif i == 1 and j == 0:
        return 1 + xg_b * rho
    elif i == 0 and j == 1:
        return 1 + xg_a * rho
    elif i == 1 and j == 1:
        return 1 - rho
    else:
        return 1.0


def score_probabilities(
    xg_a: float,
    xg_b: float,
    max_goals: int = 8,
    rho: float = DIXON_COLES_RHO,
) -> dict:
    """
    Return a dict mapping (home_goals, away_goals) -> probability.

    Probabilities are Dixon-Coles corrected and renormalised to sum to 1.

    Args:
        xg_a:      Expected goals for team A (home).
        xg_b:      Expected goals for team B (away).
        max_goals: Maximum goals per team to consider (8 is plenty —
                   P(9 goals) from xG=2.5 is less than 0.001%).
        rho:       Dixon-Coles correction strength.
    """
    probs = {}
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            raw = poisson.pmf(i, xg_a) * poisson.pmf(j, xg_b)
            probs[(i, j)] = raw * _dixon_coles_tau(i, j, xg_a, xg_b, rho)

    # Renormalise so probabilities sum to 1.0
    total = sum(probs.values())
    return {score: p / total for score, p in probs.items()}


def match_probabilities(xg_a: float, xg_b: float) -> tuple:
    """
    Return (p_home_win, p_draw, p_away_win) from expected goals.

    These are aggregated from the full scoreline probability grid.
    """
    probs = score_probabilities(xg_a, xg_b)
    p_home = sum(p for (i, j), p in probs.items() if i > j)
    p_draw  = sum(p for (i, j), p in probs.items() if i == j)
    p_away  = sum(p for (i, j), p in probs.items() if i < j)
    return round(p_home, 4), round(p_draw, 4), round(p_away, 4)


def best_prediction(
    xg_a: float,
    xg_b: float,
    points_exact: int = 25,
    points_result: int = 10,
) -> tuple:
    """
    Find the scoreline prediction that maximises expected competition points.

    This is your original EV-optimal idea from the notebook, preserved here.

    Why this matters:
        The most LIKELY scoreline is not always the best PREDICTION.
        Example: France vs Germany. Most likely score might be 1-1 (P=0.18),
        but predicting 1-0 France might earn more expected points because
        France winning is much more probable, giving you the 10-point result
        bonus across many more scenarios than just the 1-1 scoreline.

    Expected value of predicting scoreline (i, j):
        EV = P(exact score i-j) * 25
           + P(correct result but wrong score) * 10

    Args:
        points_exact:  Points awarded for correct exact scoreline.
        points_result: Points awarded for correct result only.

    Returns:
        Tuple of ((home_goals, away_goals), expected_value).
    """
    probs = score_probabilities(xg_a, xg_b)
    p_home_win = sum(p for (i, j), p in probs.items() if i > j)
    p_draw     = sum(p for (i, j), p in probs.items() if i == j)
    p_away_win = sum(p for (i, j), p in probs.items() if i < j)

    best_ev = -1.0
    best_score = (0, 0)

    for (i, j), p_exact in probs.items():
        if i > j:
            p_result = p_home_win
        elif i == j:
            p_result = p_draw
        else:
            p_result = p_away_win

        ev = p_exact * points_exact + (p_result - p_exact) * points_result
        if ev > best_ev:
            best_ev = ev
            best_score = (i, j)

    return best_score, round(best_ev, 4)
