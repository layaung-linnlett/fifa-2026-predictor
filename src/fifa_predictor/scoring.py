"""
Competition scoring module.

--- Plain English explanation ---

The DataCamp-style competition awards points for each match prediction.
Points depend on two things:
    1. HOW ACCURATE your prediction is (exact score vs correct result only)
    2. WHICH ROUND the match is in (later rounds are worth more)

POINT VALUES (before multiplier):
    Exact scoreline (e.g. predicted 2-1, actual 2-1)  -> 25 points
    Correct result only (e.g. predicted 2-1, actual 3-0) -> 10 points
    Wrong result                                         ->  0 points

ROUND MULTIPLIERS (from knockout_slots.csv):
    Group stage      x1   (matches 1-72)
    Round of 32      x1
    Round of 16      x2
    Quarter-final    x4
    Semi-final       x8
    Third-place      x8
    Final            x16

So a correct exact score in the Final is worth 25 x 16 = 400 points.
A correct exact score in the group stage is worth 25 x 1 = 25 points.

This scoring system rewards:
    - Being right about exact scores (high but rare reward)
    - Being right about results in important matches (multiplied 10 points)
    - Not wasting picks on unlikely exact scores when a result prediction
      earns nearly as many expected points
"""


# Base point values — these match the competition spec
POINTS_EXACT_SCORE = 25
POINTS_CORRECT_RESULT = 10
POINTS_WRONG = 0


def score_match(
    predicted_home: int,
    predicted_away: int,
    actual_home: int,
    actual_away: int,
    multiplier: int = 1,
) -> int:
    """
    Score a single match prediction against the actual result.

    Args:
        predicted_home: Predicted home goals.
        predicted_away: Predicted away goals.
        actual_home:    Actual home goals.
        actual_away:    Actual away goals.
        multiplier:     Round multiplier (1, 2, 4, 8, or 16).

    Returns:
        Points earned as an integer.

    Examples:
        Predicted 2-1, actual 2-1, group stage:  25 * 1 = 25 pts
        Predicted 2-1, actual 3-0, group stage:  10 * 1 = 10 pts
        Predicted 2-1, actual 0-1, group stage:   0 * 1 =  0 pts
        Predicted 2-1, actual 3-0, Final:         10 * 16 = 160 pts
    """
    pred_result = _result(predicted_home, predicted_away)
    actual_result = _result(actual_home, actual_away)

    if predicted_home == actual_home and predicted_away == actual_away:
        base = POINTS_EXACT_SCORE
    elif pred_result == actual_result:
        base = POINTS_CORRECT_RESULT
    else:
        base = POINTS_WRONG

    return base * multiplier


def score_all_matches(predictions: list[dict], actuals: list[dict]) -> dict:
    """
    Score an entire set of match predictions.

    Args:
        predictions: List of dicts, each with keys:
                     match_id, home_score, away_score, multiplier.
        actuals:     List of dicts, each with keys:
                     match_id, home_score, away_score.

    Returns:
        Dict with keys:
            "total"          -> total points across all matches
            "by_match"       -> list of dicts with per-match breakdown
            "exact_scores"   -> number of exact scores correct
            "correct_results"-> number of results correct (not exact)
            "wrong"          -> number of wrong predictions
    """
    actual_lookup = {m["match_id"]: m for m in actuals}

    total = 0
    exact_scores = 0
    correct_results = 0
    wrong = 0
    by_match = []

    for pred in predictions:
        mid = pred["match_id"]
        actual = actual_lookup.get(mid)
        if actual is None:
            continue

        multiplier = pred.get("multiplier", 1)
        pts = score_match(
            pred["home_score"], pred["away_score"],
            actual["home_score"], actual["away_score"],
            multiplier=multiplier,
        )

        total += pts
        if pts == POINTS_EXACT_SCORE * multiplier:
            exact_scores += 1
        elif pts == POINTS_CORRECT_RESULT * multiplier:
            correct_results += 1
        else:
            wrong += 1

        by_match.append({
            "match_id":      mid,
            "predicted":     f"{pred['home_score']}-{pred['away_score']}",
            "actual":        f"{actual['home_score']}-{actual['away_score']}",
            "multiplier":    multiplier,
            "points":        pts,
        })

    return {
        "total":           total,
        "by_match":        by_match,
        "exact_scores":    exact_scores,
        "correct_results": correct_results,
        "wrong":           wrong,
    }


def max_possible_score(n_matches: int, multipliers: list[int] | None = None) -> int:
    """
    Calculate the maximum achievable score if every prediction is exact.

    Useful for contextualising your actual score as a percentage of perfect.

    Args:
        n_matches:   Total number of matches predicted.
        multipliers: List of multipliers per match. If None, assumes all x1.
    """
    if multipliers is None:
        multipliers = [1] * n_matches
    return sum(POINTS_EXACT_SCORE * m for m in multipliers)


def _result(home: int, away: int) -> str:
    """Return 'H', 'D', or 'A' for the match result."""
    if home > away:
        return "H"
    elif home == away:
        return "D"
    else:
        return "A"
