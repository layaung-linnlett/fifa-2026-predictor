"""Tests for the simulation module."""

import pandas as pd
import pytest
from fifa_predictor.simulation import (
    simulate_match,
    simulate_group_stage,
    run_simulation,
    _penalty_winner,
)
from fifa_predictor.bracket import compute_group_standings


DUMMY_ELO = {
    "Spain":     1900,
    "Brazil":    1850,
    "France":    1800,
    "Germany":   1750,
    "England":   1700,
    "Argentina": 1950,
    "Morocco":   1650,
    "Japan":     1600,
}


class TestSimulateMatch:
    def test_returns_required_keys(self):
        result = simulate_match("Spain", "Brazil", DUMMY_ELO)
        assert "home_score" in result
        assert "away_score" in result
        assert "winner" in result

    def test_scores_are_non_negative(self):
        for _ in range(20):
            result = simulate_match("Spain", "Brazil", DUMMY_ELO)
            assert result["home_score"] >= 0
            assert result["away_score"] >= 0

    def test_group_draw_has_no_winner(self):
        # Force a draw by patching — run many times and check draws have winner=None
        draws_with_winner = 0
        for _ in range(200):
            result = simulate_match("Spain", "Brazil", DUMMY_ELO, knockout=False)
            if result["home_score"] == result["away_score"]:
                if result["winner"] is not None:
                    draws_with_winner += 1
        assert draws_with_winner == 0

    def test_knockout_draw_always_has_winner(self):
        # Run many times and check every result has a winner
        for _ in range(50):
            result = simulate_match("Spain", "Brazil", DUMMY_ELO, knockout=True)
            assert result["winner"] in ("Spain", "Brazil")

    def test_unknown_team_uses_default_elo(self):
        # Should not raise even if team is not in elo_ratings
        result = simulate_match("UnknownFC", "Spain", DUMMY_ELO)
        assert result["home_score"] >= 0

    def test_stronger_team_wins_more_often(self):
        # Argentina (1950) should beat Japan (1600) more than 70% of the time
        argentina_wins = sum(
            1 for _ in range(500)
            if simulate_match("Argentina", "Japan", DUMMY_ELO, knockout=True)["winner"] == "Argentina"
        )
        assert argentina_wins > 350  # > 70%


class TestPenaltyWinner:
    def test_returns_one_of_the_two_teams(self):
        for _ in range(20):
            winner = _penalty_winner("Spain", "Brazil", 1900, 1850)
            assert winner in ("Spain", "Brazil")

    def test_equal_teams_roughly_fifty_fifty(self):
        spain_wins = sum(
            1 for _ in range(1000)
            if _penalty_winner("Spain", "Brazil", 1500, 1500) == "Spain"
        )
        assert 400 < spain_wins < 600


class TestSimulateGroupStage:
    def _make_fixtures(self):
        return pd.DataFrame([
            {"match_id": 1, "group": "A", "home_team": "Spain",   "away_team": "Brazil",  "date_utc": "2026-06-11", "venue": "x"},
            {"match_id": 2, "group": "A", "home_team": "France",  "away_team": "Germany", "date_utc": "2026-06-11", "venue": "x"},
            {"match_id": 3, "group": "A", "home_team": "Spain",   "away_team": "France",  "date_utc": "2026-06-15", "venue": "x"},
            {"match_id": 4, "group": "A", "home_team": "Brazil",  "away_team": "Germany", "date_utc": "2026-06-15", "venue": "x"},
            {"match_id": 5, "group": "A", "home_team": "Spain",   "away_team": "Germany", "date_utc": "2026-06-19", "venue": "x"},
            {"match_id": 6, "group": "A", "home_team": "Brazil",  "away_team": "France",  "date_utc": "2026-06-19", "venue": "x"},
        ])

    def test_returns_standings_for_each_group(self):
        standings = simulate_group_stage(self._make_fixtures(), DUMMY_ELO)
        assert "A" in standings

    def test_each_group_has_four_teams(self):
        standings = simulate_group_stage(self._make_fixtures(), DUMMY_ELO)
        assert len(standings["A"]) == 4

    def test_standings_sorted_by_points(self):
        # Run many times — first place should always have >= points of second
        for _ in range(10):
            standings = simulate_group_stage(self._make_fixtures(), DUMMY_ELO)
            s = standings["A"]
            assert s[0]["points"] >= s[1]["points"] or s[0]["gd"] >= s[1]["gd"]


class TestRunSimulation:
    def _make_minimal_fixtures(self):
        """Minimal 1-group fixture set for fast integration testing."""
        return pd.DataFrame([
            {"match_id": i+1, "group": "A",
             "home_team": home, "away_team": away,
             "date_utc": "2026-06-11", "venue": "x"}
            for i, (home, away) in enumerate([
                ("Spain", "Brazil"), ("France", "Germany"),
                ("Spain", "France"), ("Brazil", "Germany"),
                ("Spain", "Germany"), ("Brazil", "France"),
            ])
        ])

    def _make_minimal_knockout(self):
        return pd.DataFrame([
            {"match_id": 73, "round": "Final", "multiplier": 16,
             "date_utc": "2026-07-19", "venue": "x",
             "slot_home": "Winner Group A", "slot_away": "Runner-up Group A"},
        ])

    def test_returns_dataframe(self):
        df = run_simulation(
            self._make_minimal_fixtures(),
            self._make_minimal_knockout(),
            DUMMY_ELO,
            n_sims=50,
            seed=1,
        )
        assert isinstance(df, pd.DataFrame)

    def test_title_probs_sum_to_one(self):
        df = run_simulation(
            self._make_minimal_fixtures(),
            self._make_minimal_knockout(),
            DUMMY_ELO,
            n_sims=200,
            seed=1,
        )
        assert abs(df["title_prob"].sum() - 1.0) < 0.05

    def test_stronger_team_has_higher_title_prob(self):
        df = run_simulation(
            self._make_minimal_fixtures(),
            self._make_minimal_knockout(),
            DUMMY_ELO,
            n_sims=500,
            seed=42,
        )
        # Spain (1900) should be top team in this group
        top_team = df.iloc[0]["team"]
        assert top_team == "Spain"

    def test_seed_gives_reproducible_results(self):
        kwargs = dict(
            fixtures=self._make_minimal_fixtures(),
            knockout_slots=self._make_minimal_knockout(),
            elo_ratings=DUMMY_ELO,
            n_sims=100,
            seed=7,
        )
        df1 = run_simulation(**kwargs)
        df2 = run_simulation(**kwargs)
        assert df1["title_prob"].tolist() == df2["title_prob"].tolist()
