"""
Monte Carlo tournament simulation.

--- Plain English explanation ---

A single simulation run does this:

  1. GROUP STAGE (72 matches)
     For each of the 72 group matches, we sample a random scoreline from
     the Poisson probability distribution. This is like rolling a loaded
     die — high-probability scorelines come up more often, low-probability
     ones occasionally. One run of 72 matches = one possible group stage.

  2. GROUP STANDINGS
     After all group matches are simulated, we rank teams by points,
     goal difference, goals scored (FIFA tiebreaker order). The top 2
     from each group plus the best 8 third-place teams advance.

  3. KNOCKOUT ROUNDS (Round of 32 through to Final)
     For each knockout match, we again sample a random scoreline.
     If it's a draw, we simulate a penalty shootout — the team with
     the higher Elo wins the shootout with probability 0.5 + small_edge.

  4. REPEAT 10,000 TIMES
     Each run produces a different champion, different bracket, different
     upsets. After 10,000 runs we count: how often did each team win?
     That fraction is their "title probability".

Why Monte Carlo and not just "pick the most likely winner each round"?
     The most likely winner of every match is always the stronger team.
     But that ignores the real randomness of football — any team can win
     on the day. Monte Carlo captures that uncertainty correctly.
     10,000 simulations is enough that the percentages stabilise
     (running it again gives nearly identical numbers).
"""

import random
from collections import defaultdict

import pandas as pd

from fifa_predictor.bracket import (
    GROUPS,
    compute_group_standings,
    resolve_best_third,
    build_r32_lineup,
    resolve_bracket,
)
from fifa_predictor.poisson_model import score_probabilities, expected_goals


# ---------------------------------------------------------------------------
# JUDGMENT CALL: penalty shootout edge
#
# In a real shootout, both teams score roughly 75% of penalties. The better
# team has a slight edge — we model this as a small Elo-based tilt on top
# of 50/50. MAX_PENALTY_EDGE = 0.1 means the strongest possible team has
# at most a 60% chance of winning a shootout vs any opponent.
# ---------------------------------------------------------------------------
MAX_PENALTY_EDGE = 0.1


def _sample_score(xg_a: float, xg_b: float) -> tuple:
    """
    Sample a single scoreline from the Poisson distribution.

    This is the core randomness engine. We compute the full probability
    grid, then pick one scoreline at random weighted by those probabilities.
    Think of it like spinning a roulette wheel where each pocket is a
    different scoreline sized by its probability.
    """
    probs = score_probabilities(xg_a, xg_b)
    scores = list(probs.keys())
    weights = list(probs.values())
    return random.choices(scores, weights=weights, k=1)[0]


def _penalty_winner(team_a: str, team_b: str, elo_a: float, elo_b: float) -> str:
    """
    Simulate a penalty shootout. Returns the winning team name.

    The team with the higher Elo gets a small edge (up to MAX_PENALTY_EDGE
    above 50%). This reflects that better teams tend to handle pressure
    slightly better, but shootouts are still largely 50/50.
    """
    elo_diff = elo_a - elo_b
    # Clamp edge so it never exceeds MAX_PENALTY_EDGE
    edge = max(-MAX_PENALTY_EDGE, min(MAX_PENALTY_EDGE, elo_diff / 4000))
    p_a_wins = 0.5 + edge
    return team_a if random.random() < p_a_wins else team_b


def simulate_match(
    team_a: str,
    team_b: str,
    elo_ratings: dict,
    knockout: bool = False,
) -> dict:
    """
    Simulate one match and return the result.

    Args:
        team_a:      Home / first team.
        team_b:      Away / second team.
        elo_ratings: Dict of team -> adjusted Elo rating.
        knockout:    If True, draw is resolved by penalty shootout.

    Returns:
        Dict with keys: home_score, away_score, winner (or None for group draws).
    """
    elo_a = elo_ratings.get(team_a, 1500)
    elo_b = elo_ratings.get(team_b, 1500)
    xg_a, xg_b = expected_goals(elo_a, elo_b)
    home_score, away_score = _sample_score(xg_a, xg_b)

    if home_score > away_score:
        winner = team_a
    elif away_score > home_score:
        winner = team_b
    elif knockout:
        winner = _penalty_winner(team_a, team_b, elo_a, elo_b)
    else:
        winner = None  # draw in group stage, no winner declared

    return {
        "home_team": team_a,
        "away_team": team_b,
        "home_score": home_score,
        "away_score": away_score,
        "winner": winner,
    }


def simulate_group_stage(
    fixtures: pd.DataFrame,
    elo_ratings: dict,
) -> dict:
    """
    Simulate all 72 group matches and return standings for each group.

    Args:
        fixtures:    DataFrame from group_fixtures.csv.
        elo_ratings: Adjusted Elo ratings dict.

    Returns:
        Dict mapping group letter -> sorted standings list.
    """
    group_results = defaultdict(list)

    for _, row in fixtures.iterrows():
        result = simulate_match(row["home_team"], row["away_team"], elo_ratings)
        group_results[row["group"]].append(result)

    all_standings = {}
    for grp in GROUPS:
        matches = group_results.get(grp, [])
        all_standings[grp] = compute_group_standings(grp, matches)

    return all_standings


def simulate_knockout_stage(
    knockout_slots: pd.DataFrame,
    r32_lineup: dict,
    elo_ratings: dict,
) -> dict:
    """
    Simulate the full knockout bracket from Round of 32 to Final.

    Works by processing matches in match_id order. For each match,
    we resolve the slot strings to actual team names (using results
    already accumulated), then simulate the match.

    Args:
        knockout_slots: DataFrame from knockout_slots.csv.
        r32_lineup:     Output of build_r32_lineup() — maps slot -> team.
        elo_ratings:    Adjusted Elo ratings dict.

    Returns:
        Dict with keys:
            "winners"  -> {match_id: team_name}
            "losers"   -> {match_id: team_name}
            "champion" -> team_name
            "finalist" -> team_name (runner-up)
            "third"    -> team_name (third place)
            "semi_losers" -> [team_name, team_name]
    """
    match_winners = {}
    match_losers = {}

    # Sort by match_id to process rounds in order
    sorted_matches = knockout_slots.sort_values("match_id")

    champion = None
    finalist = None
    third = None
    semi_losers = []

    for _, row in sorted_matches.iterrows():
        match_id = int(row["match_id"])
        round_name = row["round"]

        # Resolve slot strings to team names
        resolved = resolve_bracket(
            knockout_slots, r32_lineup, match_winners, match_losers
        )

        home_slot = row["slot_home"]
        away_slot = row["slot_away"]

        team_a = resolved.get(home_slot)
        team_b = resolved.get(away_slot)

        # If either team can't be resolved (e.g. missing best-third slot),
        # use a placeholder so the simulation doesn't crash
        if team_a is None:
            team_a = f"TBD_{home_slot}"
        if team_b is None:
            team_b = f"TBD_{away_slot}"

        is_third_place = "Third" in round_name
        result = simulate_match(team_a, team_b, elo_ratings, knockout=True)

        match_winners[match_id] = result["winner"]
        loser = team_b if result["winner"] == team_a else team_a
        match_losers[match_id] = loser

        # Track final and third-place outcomes
        if round_name == "Final":
            champion = result["winner"]
            finalist = loser
        elif is_third_place:
            third = result["winner"]
        elif round_name == "Semi-final":
            semi_losers.append(loser)

    return {
        "winners": match_winners,
        "losers": match_losers,
        "champion": champion,
        "finalist": finalist,
        "third": third,
        "semi_losers": semi_losers,
    }


def run_simulation(
    fixtures: pd.DataFrame,
    knockout_slots: pd.DataFrame,
    elo_ratings: dict,
    n_sims: int = 10_000,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Run the full Monte Carlo simulation n_sims times.

    Args:
        fixtures:       Group stage fixtures DataFrame.
        knockout_slots: Knockout bracket DataFrame.
        elo_ratings:    Adjusted Elo ratings (from elo.apply_form_adjustment).
        n_sims:         Number of simulations to run.
        seed:           Random seed for reproducibility. Pass None for
                        different results each run.

    Returns:
        DataFrame with one row per team, columns:
            team, title_prob, finalist_prob, semi_prob, third_prob
        Sorted by title_prob descending.

    JUDGMENT CALL — seed=42:
        Using a fixed seed means results are reproducible — running the
        script twice gives identical numbers. This is important for a
        portfolio project (you can screenshot results and they won't change).
        Pass seed=None if you want genuine random variation between runs.
    """
    if seed is not None:
        random.seed(seed)

    # Counters for each team's outcomes across all simulations
    titles      = defaultdict(int)
    finals      = defaultdict(int)
    semis       = defaultdict(int)
    thirds      = defaultdict(int)

    for _ in range(n_sims):
        # --- Group stage ---
        all_standings = simulate_group_stage(fixtures, elo_ratings)
        r32_lineup = build_r32_lineup(all_standings)

        # Count semi-final appearances (all R32 qualifiers reached the groups)
        # We track from knockout stage onwards
        ko_result = simulate_knockout_stage(knockout_slots, r32_lineup, elo_ratings)

        if ko_result["champion"]:
            titles[ko_result["champion"]] += 1
        if ko_result["finalist"]:
            finals[ko_result["finalist"]] += 1
        if ko_result["third"]:
            thirds[ko_result["third"]] += 1
        for team in ko_result["semi_losers"]:
            semis[team] += 1

    # Collect all teams that appeared in any outcome
    all_teams = set(titles) | set(finals) | set(semis) | set(thirds)

    rows = []
    for team in all_teams:
        rows.append({
            "team":          team,
            "title_prob":    round(titles[team] / n_sims, 4),
            "finalist_prob": round((titles[team] + finals[team]) / n_sims, 4),
            "semi_prob":     round((titles[team] + finals[team] + semis[team]) / n_sims, 4),
            "third_prob":    round(thirds[team] / n_sims, 4),
        })

    df = pd.DataFrame(rows).sort_values("title_prob", ascending=False).reset_index(drop=True)
    return df
