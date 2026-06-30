"""
Command-line interface for the FIFA 2026 prediction engine.

Why click over argparse?
    Both work fine. click is chosen here because:
    - Commands look like `fifa-predictor simulate --sims 10000` rather than
      subparser boilerplate — cleaner for a portfolio README.
    - `@click.command` decorators keep each command self-contained and easy
      to read — you can understand one command without reading the others.
    - Automatic --help text generation is nicer out of the box.
    For a junior portfolio project, either is fine. click is the industry
    preference for user-facing CLIs; argparse is fine for internal scripts.

Usage (after pip install -e .):
    python -m fifa_predictor predict-groups
    python -m fifa_predictor predict-groups --out outputs/predictions/group_predictions.csv
    python -m fifa_predictor simulate --sims 10000
    python -m fifa_predictor predict-knockout
    python -m fifa_predictor predict-knockout --out outputs/predictions/knockout_predictions.csv
"""

import os
import click
import pandas as pd

from fifa_predictor.elo import compute_elo_ratings, apply_form_adjustment
from fifa_predictor.poisson_model import expected_goals, best_prediction, match_probabilities
from fifa_predictor.cards_corners import predict_corners_cards
from fifa_predictor.bracket import build_r32_lineup, compute_group_standings, GROUPS
from fifa_predictor.simulation import run_simulation, simulate_group_stage


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _data_path(filename: str) -> str:
    """Resolve a data file path relative to the project root."""
    here = os.path.dirname(__file__)
    return os.path.join(here, "..", "..", "data", "raw", filename)


def _load_data() -> tuple:
    """Load all raw data files. Raises a clear error if any are missing."""
    results_path  = _data_path("results.csv")
    fixtures_path = _data_path("group_fixtures.csv")
    knockout_path = _data_path("knockout_slots.csv")

    for path in [results_path, fixtures_path, knockout_path]:
        if not os.path.exists(path):
            raise click.ClickException(f"Data file not found: {path}")

    results  = pd.read_csv(results_path)
    fixtures = pd.read_csv(fixtures_path)
    knockout = pd.read_csv(knockout_path)
    return results, fixtures, knockout


def _build_elo(results: pd.DataFrame, start_date: str) -> dict:
    """Compute base Elo ratings and apply form adjustment."""
    click.echo(f"  Computing Elo ratings from {start_date}...")
    base = compute_elo_ratings(results, start_date=start_date)
    adj  = apply_form_adjustment(base, results)
    click.echo(f"  Rated {len(adj)} teams.")
    return adj


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def main():
    """FIFA World Cup 2026 prediction engine."""
    pass


# ---------------------------------------------------------------------------
# Command: predict-groups
# ---------------------------------------------------------------------------

@main.command("predict-groups")
@click.option(
    "--out", default="outputs/predictions/group_predictions.csv",
    show_default=True, help="Output CSV path."
)
@click.option(
    "--start-date", default="2014-01-01",
    show_default=True, help="Earliest historical match date to use for Elo."
)
def predict_groups(out: str, start_date: str):
    """
    Predict all 72 group-stage matches and save to CSV.

    Uses the EV-optimal scoreline (maximises expected competition points)
    rather than the raw most-likely score. Also outputs win/draw/loss
    probabilities and corners/cards estimates per match.
    """
    click.echo("=== predict-groups ===")
    results, fixtures, _ = _load_data()
    elo = _build_elo(results, start_date)

    rows = []
    for _, row in fixtures.iterrows():
        home, away = row["home_team"], row["away_team"]
        xg_h, xg_a = expected_goals(elo.get(home, 1500), elo.get(away, 1500))
        score, ev   = best_prediction(xg_h, xg_a)
        hw, d, aw   = match_probabilities(xg_h, xg_a)
        c, y, r     = predict_corners_cards(home, away)

        rows.append({
            "match_id":   row["match_id"],
            "group":      row["group"],
            "home_team":  home,
            "away_team":  away,
            "home_score": score[0],
            "away_score": score[1],
            "expected_ev": ev,
            "home_win%":  round(hw * 100, 1),
            "draw%":      round(d  * 100, 1),
            "away_win%":  round(aw * 100, 1),
            "corners":    c,
            "yellows":    y,
            "reds":       r,
        })

    os.makedirs(os.path.dirname(out), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    click.echo(f"  Saved {len(df)} match predictions -> {out}")


# ---------------------------------------------------------------------------
# Command: simulate
# ---------------------------------------------------------------------------

@main.command("simulate")
@click.option(
    "--sims", default=10_000, show_default=True,
    help="Number of Monte Carlo simulations to run."
)
@click.option(
    "--seed", default=42, show_default=True,
    help="Random seed. Use 0 for a different result each run."
)
@click.option(
    "--start-date", default="2014-01-01",
    show_default=True, help="Earliest historical match date to use for Elo."
)
@click.option(
    "--out", default="outputs/predictions/title_odds.csv",
    show_default=True, help="Output CSV path."
)
def simulate(sims: int, seed: int, start_date: str, out: str):
    """
    Run Monte Carlo simulation and output title/finalist/semi odds per team.

    Simulates the full tournament (group stage + knockout) the given number
    of times. Each run samples random scorelines from the Poisson model,
    so upsets happen at the right frequency. Aggregates title, finalist,
    and semi-final probabilities per team.
    """
    click.echo("=== simulate ===")
    results, fixtures, knockout = _load_data()
    elo = _build_elo(results, start_date)

    actual_seed = None if seed == 0 else seed
    click.echo(f"  Running {sims:,} simulations (seed={actual_seed})...")

    df = run_simulation(fixtures, knockout, elo, n_sims=sims, seed=actual_seed)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    click.echo(f"\n  Top 10 title contenders ({sims:,} simulations):")
    click.echo(f"  {'Team':<20} {'Title%':>8} {'Final%':>8} {'Semi%':>8}")
    click.echo("  " + "-" * 48)
    for _, row in df.head(10).iterrows():
        click.echo(
            f"  {row['team']:<20}"
            f" {row['title_prob']*100:>7.1f}%"
            f" {row['finalist_prob']*100:>7.1f}%"
            f" {row['semi_prob']*100:>7.1f}%"
        )
    click.echo(f"\n  Full results saved -> {out}")


# ---------------------------------------------------------------------------
# Command: predict-knockout
# ---------------------------------------------------------------------------

@main.command("predict-knockout")
@click.option(
    "--sims", default=10_000, show_default=True,
    help="Simulations used to determine most likely bracket path."
)
@click.option(
    "--seed", default=42, show_default=True,
    help="Random seed."
)
@click.option(
    "--start-date", default="2014-01-01",
    show_default=True, help="Earliest historical match date to use for Elo."
)
@click.option(
    "--out", default="outputs/predictions/knockout_predictions.csv",
    show_default=True, help="Output CSV path."
)
def predict_knockout(sims: int, seed: int, start_date: str, out: str):
    """
    Predict the full knockout bracket and save to CSV.

    Strategy: run the group stage simulation to determine which teams
    most commonly qualify, then use the EV-optimal scoreline for each
    knockout match based on those teams' Elo ratings.

    The 'most likely bracket' is assembled from the team that appeared
    most often in each slot across all simulations.
    """
    click.echo("=== predict-knockout ===")
    results, fixtures, knockout = _load_data()
    elo = _build_elo(results, start_date)

    import random
    actual_seed = None if seed == 0 else seed
    if actual_seed is not None:
        random.seed(actual_seed)

    click.echo(f"  Simulating group stage {sims:,} times to find likely qualifiers...")

    # Count how often each team appears in each slot across many simulations
    from collections import defaultdict, Counter
    slot_counts: dict[str, Counter] = defaultdict(Counter)

    for _ in range(sims):
        all_standings = simulate_group_stage(fixtures, elo)
        lineup = build_r32_lineup(all_standings)
        for slot, team in lineup.items():
            slot_counts[slot][team] += 1

    # Most likely team for each slot
    most_likely_lineup = {
        slot: counter.most_common(1)[0][0]
        for slot, counter in slot_counts.items()
    }

    # Now predict knockout matches using EV-optimal scorelines
    from fifa_predictor.bracket import resolve_bracket
    match_winners = {}
    match_losers  = {}
    rows = []

    for _, row in knockout.sort_values("match_id").iterrows():
        match_id   = int(row["match_id"])
        round_name = row["round"]
        multiplier = int(row["multiplier"])

        resolved = resolve_bracket(knockout, most_likely_lineup, match_winners, match_losers)

        home_slot = row["slot_home"]
        away_slot = row["slot_away"]
        team_a = resolved.get(home_slot, f"TBD ({home_slot})")
        team_b = resolved.get(away_slot, f"TBD ({away_slot})")

        xg_h, xg_a = expected_goals(elo.get(team_a, 1500), elo.get(team_b, 1500))
        score, ev   = best_prediction(xg_h, xg_a)
        hw, d, aw   = match_probabilities(xg_h, xg_a)
        c, y, r     = predict_corners_cards(team_a, team_b)

        # Winner for bracket progression
        if score[0] > score[1]:
            winner = team_a
            loser  = team_b
        elif score[1] > score[0]:
            winner = team_b
            loser  = team_a
        else:
            # Predicted draw in knockout — give to higher Elo team
            winner = team_a if elo.get(team_a, 1500) >= elo.get(team_b, 1500) else team_b
            loser  = team_b if winner == team_a else team_a

        match_winners[match_id] = winner
        match_losers[match_id]  = loser

        rows.append({
            "match_id":    match_id,
            "round":       round_name,
            "multiplier":  multiplier,
            "home_team":   team_a,
            "away_team":   team_b,
            "home_score":  score[0],
            "away_score":  score[1],
            "predicted_winner": winner,
            "expected_ev": ev,
            "home_win%":   round(hw * 100, 1),
            "draw%":       round(d  * 100, 1),
            "away_win%":   round(aw * 100, 1),
            "corners":     c,
            "yellows":     y,
            "reds":        r,
        })

    os.makedirs(os.path.dirname(out), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    # Print a readable bracket summary
    click.echo("\n  Predicted bracket:")
    for rnd in ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Third-place playoff", "Final"]:
        rnd_matches = df[df["round"] == rnd]
        if rnd_matches.empty:
            continue
        click.echo(f"\n  --- {rnd} ---")
        for _, m in rnd_matches.iterrows():
            click.echo(f"  {m['home_team']:<22} {int(m['home_score'])}-{int(m['away_score'])}  {m['away_team']}")

    click.echo(f"\n  Full bracket saved -> {out}")


if __name__ == "__main__":
    main()
