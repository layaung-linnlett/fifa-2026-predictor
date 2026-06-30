"""
Elo rating system for international football.

How Elo works (plain English):
    Every team has a number (their rating). Before a match, we calculate how
    surprising the result was — if a strong team beats a weak team, that's
    expected, so ratings barely move. If the weak team wins, that's a big
    surprise and ratings shift a lot. After enough matches the ratings settle
    into a stable ranking that reflects real-world team strength.

Improvements over the prototype notebook:
    1. Match importance weights  — a World Cup match moves ratings more than a
       friendly, because it's a better signal of true team strength.
    2. Cleaner separation       — compute_elo_ratings() is one pure function
       that takes a DataFrame and returns a dict. Easy to test.
    3. Form adjustment kept     — your recency-weighted form score nudges the
       rating up/down by up to 50 points before a prediction is made.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# JUDGMENT CALL: K-factors by tournament importance
#
# K controls how much a single result shifts the rating. Higher K = ratings
# react faster to new results but become noisier. The values below follow
# the same philosophy as the World Football Elo Ratings (eloratings.net),
# which is the standard reference for this kind of model.
#
# Key numbers to know for an interview:
#   - Friendlies get K=20 because teams often rest players / experiment.
#   - World Cup matches get K=60 because they are the highest-stakes signal.
#   - Everything else sits in between.
# ---------------------------------------------------------------------------
IMPORTANCE_K = {
    "FIFA World Cup":               60,
    "Confederations Cup":           50,
    "Copa America":                 45,
    "Copa América":                 45,
    "UEFA Euro":                    45,
    "Africa Cup of Nations":        45,
    "AFC Asian Cup":                45,
    "CONCACAF Gold Cup":            45,
    "Copa América Centenario":      45,
    "UEFA Nations League":          40,
    "FIFA World Cup qualification": 35,
    "UEFA Euro qualification":      35,
    "Friendly":                     20,
}
DEFAULT_K = 30  # for any tournament not in the dict above


def _get_k(tournament: str) -> float:
    """Return the K-factor for a given tournament string.

    Matches longest key first so 'FIFA World Cup qualification' (K=35)
    is not swallowed by the shorter 'FIFA World Cup' (K=60).
    """
    t = tournament.lower()
    for key in sorted(IMPORTANCE_K, key=len, reverse=True):
        if key.lower() in t:
            return IMPORTANCE_K[key]
    return DEFAULT_K


def update_elo(
    rating_a: float,
    rating_b: float,
    result_a: float,
    k: float = 30,
) -> tuple:
    """
    Apply one Elo update and return new (rating_a, rating_b).

    Args:
        rating_a: Current Elo of team A (home team).
        rating_b: Current Elo of team B (away team).
        result_a: Actual outcome for team A: 1.0=win, 0.5=draw, 0.0=loss.
        k:        K-factor (how much this match moves the needle).

    The maths:
        expected_a = 1 / (1 + 10^((rating_b - rating_a) / 400))
        new_rating_a = rating_a + K * (result_a - expected_a)

    The 400 is a scaling constant — a team rated 400 points higher is
    expected to win roughly 91% of the time.
    """
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    new_a = rating_a + k * (result_a - expected_a)
    new_b = rating_b + k * ((1 - result_a) - (1 - expected_a))
    return round(new_a, 1), round(new_b, 1)


def compute_elo_ratings(
    results: pd.DataFrame,
    start_date: str = "2014-01-01",
    default_rating: float = 1500.0,
) -> dict:
    """
    Compute Elo ratings for all teams from historical match results.

    Args:
        results:        DataFrame with columns: date, home_team, away_team,
                        home_score, away_score, tournament.
        start_date:     Only use matches on or after this date.
        default_rating: Starting rating for any team with no prior history.

    Returns:
        Dict mapping team name -> final Elo rating.

    JUDGMENT CALL — why 2014?
        Using all 49k matches back to 1872 sounds thorough but hurts the model.
        Teams from 1920 are essentially different nations with different players
        and tactics. 2014 (post-Brazil World Cup) gives ~12 years of modern
        football, covering every current squad's competitive generation.
        You can adjust start_date via the CLI if you want to experiment.
    """
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= start_date].sort_values("date")
    df = df.dropna(subset=["home_score", "away_score"])

    ratings = {}

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        r_home = ratings.get(home, default_rating)
        r_away = ratings.get(away, default_rating)

        if row["home_score"] > row["away_score"]:
            result = 1.0
        elif row["home_score"] == row["away_score"]:
            result = 0.5
        else:
            result = 0.0

        k = _get_k(str(row.get("tournament", "")))
        ratings[home], ratings[away] = update_elo(r_home, r_away, result, k)

    return ratings


def get_form_score(team: str, results: pd.DataFrame, n: int = 10) -> float:
    """
    Return a 0-1 form score based on the team's last n completed matches.

    0.5 = average (wins and losses balanced).
    > 0.5 = good recent form.
    < 0.5 = poor recent form.

    This is your original idea from the notebook, kept as a standalone
    function so it can be unit-tested independently.
    """
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    recent = (
        df[mask]
        .dropna(subset=["home_score", "away_score"])
        .sort_values("date")
        .tail(n)
    )

    if recent.empty:
        return 0.5

    points = []
    for _, row in recent.iterrows():
        if row["home_team"] == team:
            if row["home_score"] > row["away_score"]:
                pts = 1.0
            elif row["home_score"] == row["away_score"]:
                pts = 0.5
            else:
                pts = 0.0
        else:
            if row["away_score"] > row["home_score"]:
                pts = 1.0
            elif row["away_score"] == row["home_score"]:
                pts = 0.5
            else:
                pts = 0.0
        points.append(pts)

    return sum(points) / len(points)


def apply_form_adjustment(
    ratings: dict,
    results: pd.DataFrame,
    n: int = 10,
    weight: float = 100.0,
) -> dict:
    """
    Nudge each team's Elo rating up or down based on recent form.

        adjusted = base_elo + (form_score - 0.5) * weight

    With weight=100: a team on a perfect winning streak gets +50 points,
    a team that lost every recent match gets -50 points.

    JUDGMENT CALL — weight=100:
        This is a modest nudge. At 100, even perfect form only shifts by 50
        Elo points, which changes a win probability by about 2 percentage
        points. You could raise it to 150 to amplify recent form more
        aggressively, but 100 keeps it from dominating the base Elo signal.
    """
    adjusted = {}
    for team, rating in ratings.items():
        form = get_form_score(team, results, n=n)
        adjusted[team] = round(rating + (form - 0.5) * weight, 1)
    return adjusted
