"""Tests for the Poisson scoring model."""

import pytest
from fifa_predictor.poisson_model import (
    win_probability,
    expected_goals,
    score_probabilities,
    match_probabilities,
    best_prediction,
    _dixon_coles_tau,
)


class TestWinProbability:
    def test_equal_teams_fifty_fifty(self):
        assert abs(win_probability(1500, 1500) - 0.5) < 0.001

    def test_higher_elo_wins_more(self):
        assert win_probability(1800, 1400) > 0.5

    def test_lower_elo_wins_less(self):
        assert win_probability(1400, 1800) < 0.5

    def test_probabilities_are_complementary(self):
        p = win_probability(1700, 1500)
        q = win_probability(1500, 1700)
        assert abs(p + q - 1.0) < 0.001


class TestExpectedGoals:
    def test_equal_teams_get_base_goals(self):
        xg_a, xg_b = expected_goals(1500, 1500, base=1.3)
        assert abs(xg_a - 1.3) < 0.01
        assert abs(xg_b - 1.3) < 0.01

    def test_stronger_team_gets_more_xg(self):
        xg_a, xg_b = expected_goals(1800, 1400)
        assert xg_a > xg_b

    def test_xg_values_are_positive(self):
        xg_a, xg_b = expected_goals(2000, 1000)
        assert xg_a > 0
        assert xg_b > 0


class TestDixonColesTau:
    def test_no_correction_for_high_scores(self):
        assert _dixon_coles_tau(3, 2, 1.5, 1.2, 0.1) == 1.0
        assert _dixon_coles_tau(2, 0, 1.5, 1.2, 0.1) == 1.0

    def test_zero_zero_reduced(self):
        # tau for 0-0 should be less than 1 (probability reduced)
        tau = _dixon_coles_tau(0, 0, 1.3, 1.3, 0.1)
        assert tau < 1.0

    def test_one_zero_increased(self):
        # tau for 1-0 should be greater than 1 (probability increased)
        tau = _dixon_coles_tau(1, 0, 1.3, 1.3, 0.1)
        assert tau > 1.0

    def test_zero_one_increased(self):
        tau = _dixon_coles_tau(0, 1, 1.3, 1.3, 0.1)
        assert tau > 1.0

    def test_one_one_reduced(self):
        tau = _dixon_coles_tau(1, 1, 1.3, 1.3, 0.1)
        assert tau < 1.0

    def test_rho_zero_means_no_correction(self):
        # With rho=0, tau should equal 1.0 for all low scores
        for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            assert _dixon_coles_tau(i, j, 1.3, 1.3, rho=0.0) == 1.0


class TestScoreProbabilities:
    def test_probabilities_sum_to_one(self):
        probs = score_probabilities(1.5, 1.2)
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_all_probabilities_non_negative(self):
        probs = score_probabilities(1.5, 1.2)
        assert all(p >= 0 for p in probs.values())

    def test_grid_size(self):
        probs = score_probabilities(1.5, 1.2, max_goals=8)
        assert len(probs) == 9 * 9  # 0..8 for each team

    def test_dominant_team_wins_more_often(self):
        probs = score_probabilities(3.0, 0.3)
        p_home_win = sum(p for (i, j), p in probs.items() if i > j)
        p_away_win = sum(p for (i, j), p in probs.items() if i < j)
        assert p_home_win > p_away_win


class TestMatchProbabilities:
    def test_probabilities_sum_to_one(self):
        p_h, p_d, p_a = match_probabilities(1.5, 1.2)
        assert abs(p_h + p_d + p_a - 1.0) < 1e-3

    def test_equal_teams_symmetric(self):
        p_h, p_d, p_a = match_probabilities(1.3, 1.3)
        assert abs(p_h - p_a) < 0.01

    def test_stronger_team_more_likely_to_win(self):
        p_h, _, p_a = match_probabilities(2.5, 0.5)
        assert p_h > p_a


class TestBestPrediction:
    def test_returns_tuple_of_score_and_ev(self):
        score, ev = best_prediction(2.0, 0.5)
        assert isinstance(score, tuple)
        assert len(score) == 2
        assert isinstance(ev, float)

    def test_ev_is_positive(self):
        _, ev = best_prediction(1.5, 1.2)
        assert ev > 0

    def test_dominant_team_predicted_to_win(self):
        score, _ = best_prediction(3.0, 0.3)
        home_goals, away_goals = score
        assert home_goals > away_goals

    def test_higher_exact_points_shifts_prediction(self):
        # With very high exact-score reward, best prediction = most likely score
        score_high, _ = best_prediction(1.5, 1.2, points_exact=1000, points_result=1)
        probs = score_probabilities(1.5, 1.2)
        most_likely = max(probs, key=probs.get)
        assert score_high == most_likely
