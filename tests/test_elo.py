"""Tests for the Elo rating module."""

import pandas as pd
import pytest
from fifa_predictor.elo import (
    update_elo,
    compute_elo_ratings,
    get_form_score,
    apply_form_adjustment,
    _get_k,
)


class TestUpdateElo:
    def test_winner_gains_points(self):
        new_a, new_b = update_elo(1500, 1500, result_a=1.0)
        assert new_a > 1500
        assert new_b < 1500

    def test_loser_loses_points(self):
        new_a, new_b = update_elo(1500, 1500, result_a=0.0)
        assert new_a < 1500
        assert new_b > 1500

    def test_draw_equal_teams_no_change(self):
        new_a, new_b = update_elo(1500, 1500, result_a=0.5)
        assert new_a == 1500.0
        assert new_b == 1500.0

    def test_ratings_sum_is_conserved(self):
        """Total Elo in the system must stay constant after any result."""
        for result in [0.0, 0.5, 1.0]:
            a, b = update_elo(1600, 1400, result_a=result)
            assert abs((a + b) - (1600 + 1400)) < 0.1

    def test_upset_moves_ratings_more_than_expected_result(self):
        """Beating a much stronger team should move ratings more."""
        # Weak team (1200) beats strong team (1800) — big upset
        _, b_upset = update_elo(1200, 1800, result_a=1.0, k=30)
        # Strong team beats weak team — expected result
        a_expected, _ = update_elo(1800, 1200, result_a=1.0, k=30)
        upset_gain = b_upset - 1800   # strong team lost points
        expected_gain = a_expected - 1800  # strong team gained points
        assert abs(upset_gain) > expected_gain


class TestGetK:
    def test_world_cup_has_highest_k(self):
        assert _get_k("FIFA World Cup") == 60

    def test_friendly_has_lowest_k(self):
        assert _get_k("Friendly") == 20

    def test_unknown_tournament_returns_default(self):
        assert _get_k("Some Random Cup") == 30

    def test_case_insensitive_match(self):
        assert _get_k("fifa world cup qualification") == 35


class TestComputeEloRatings:
    def _make_results(self):
        return pd.DataFrame({
            "date": ["2020-01-01", "2020-06-01", "2021-01-01"],
            "home_team": ["England", "England", "France"],
            "away_team": ["France",  "Germany",  "Germany"],
            "home_score": [2, 1, 0],
            "away_score": [1, 0, 1],
            "tournament": ["Friendly", "UEFA Euro", "Friendly"],
        })

    def test_returns_dict(self):
        ratings = compute_elo_ratings(self._make_results(), start_date="2019-01-01")
        assert isinstance(ratings, dict)

    def test_all_teams_present(self):
        ratings = compute_elo_ratings(self._make_results(), start_date="2019-01-01")
        assert "England" in ratings
        assert "France" in ratings
        assert "Germany" in ratings

    def test_start_date_filters_matches(self):
        ratings_all = compute_elo_ratings(self._make_results(), start_date="2019-01-01")
        ratings_late = compute_elo_ratings(self._make_results(), start_date="2021-01-01")
        # With fewer matches, ratings should differ
        assert ratings_all != ratings_late

    def test_winner_ends_up_higher(self):
        """England wins both its matches — should end up above 1500."""
        ratings = compute_elo_ratings(self._make_results(), start_date="2019-01-01")
        assert ratings["England"] > 1500


class TestFormScore:
    def _make_results(self, wins=7, draws=2, losses=1):
        rows = []
        for i in range(wins):
            rows.append({"date": f"2024-01-{i+1:02d}", "home_team": "England",
                         "away_team": "Other", "home_score": 2, "away_score": 0,
                         "tournament": "Friendly"})
        for i in range(draws):
            rows.append({"date": f"2024-02-{i+1:02d}", "home_team": "England",
                         "away_team": "Other", "home_score": 1, "away_score": 1,
                         "tournament": "Friendly"})
        for i in range(losses):
            rows.append({"date": f"2025-01-{i+1:02d}", "home_team": "England",
                         "away_team": "Other", "home_score": 0, "away_score": 1,
                         "tournament": "Friendly"})
        return pd.DataFrame(rows)

    def test_perfect_form_returns_one(self):
        df = self._make_results(wins=10, draws=0, losses=0)
        assert get_form_score("England", df) == 1.0

    def test_no_wins_returns_zero(self):
        df = self._make_results(wins=0, draws=0, losses=10)
        assert get_form_score("England", df) == 0.0

    def test_unknown_team_returns_half(self):
        df = self._make_results()
        assert get_form_score("NonExistentFC", df) == 0.5

    def test_form_between_zero_and_one(self):
        df = self._make_results()
        score = get_form_score("England", df)
        assert 0.0 <= score <= 1.0
