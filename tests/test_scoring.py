"""Tests for the scoring module."""

import pytest
from fifa_predictor.scoring import (
    score_match,
    score_all_matches,
    max_possible_score,
    _result,
    POINTS_EXACT_SCORE,
    POINTS_CORRECT_RESULT,
)


class TestResult:
    def test_home_win(self):
        assert _result(2, 0) == "H"

    def test_draw(self):
        assert _result(1, 1) == "D"

    def test_away_win(self):
        assert _result(0, 1) == "A"


class TestScoreMatch:
    def test_exact_score_group_stage(self):
        assert score_match(2, 1, 2, 1, multiplier=1) == 25

    def test_correct_result_only(self):
        assert score_match(2, 1, 3, 0, multiplier=1) == 10

    def test_wrong_result(self):
        assert score_match(2, 1, 0, 1, multiplier=1) == 0

    def test_multiplier_applied_to_exact(self):
        assert score_match(2, 1, 2, 1, multiplier=16) == 400

    def test_multiplier_applied_to_result(self):
        assert score_match(2, 1, 3, 0, multiplier=16) == 160

    def test_multiplier_applied_to_wrong(self):
        assert score_match(2, 1, 0, 1, multiplier=16) == 0

    def test_draw_prediction_exact(self):
        assert score_match(1, 1, 1, 1, multiplier=1) == 25

    def test_draw_prediction_correct_result(self):
        assert score_match(1, 1, 0, 0, multiplier=1) == 10

    def test_predicted_draw_wrong_when_home_wins(self):
        assert score_match(1, 1, 2, 0, multiplier=1) == 0


class TestScoreAllMatches:
    def _make_data(self):
        predictions = [
            {"match_id": 1, "home_score": 2, "away_score": 1, "multiplier": 1},
            {"match_id": 2, "home_score": 1, "away_score": 1, "multiplier": 2},
            {"match_id": 3, "home_score": 0, "away_score": 1, "multiplier": 4},
        ]
        actuals = [
            {"match_id": 1, "home_score": 2, "away_score": 1},  # exact
            {"match_id": 2, "home_score": 0, "away_score": 0},  # correct result (draw)
            {"match_id": 3, "home_score": 2, "away_score": 0},  # wrong
        ]
        return predictions, actuals

    def test_total_points(self):
        preds, acts = self._make_data()
        result = score_all_matches(preds, acts)
        # match1: 25*1=25, match2: 10*2=20, match3: 0
        assert result["total"] == 45

    def test_exact_score_count(self):
        preds, acts = self._make_data()
        result = score_all_matches(preds, acts)
        assert result["exact_scores"] == 1

    def test_correct_result_count(self):
        preds, acts = self._make_data()
        result = score_all_matches(preds, acts)
        assert result["correct_results"] == 1

    def test_wrong_count(self):
        preds, acts = self._make_data()
        result = score_all_matches(preds, acts)
        assert result["wrong"] == 1

    def test_by_match_has_correct_length(self):
        preds, acts = self._make_data()
        result = score_all_matches(preds, acts)
        assert len(result["by_match"]) == 3

    def test_missing_actual_skipped(self):
        preds = [{"match_id": 99, "home_score": 1, "away_score": 0, "multiplier": 1}]
        acts = []
        result = score_all_matches(preds, acts)
        assert result["total"] == 0


class TestMaxPossibleScore:
    def test_all_group_stage(self):
        assert max_possible_score(72) == 72 * 25

    def test_with_multipliers(self):
        multipliers = [1, 2, 4, 16]
        assert max_possible_score(4, multipliers) == (1 + 2 + 4 + 16) * 25

    def test_full_tournament(self):
        multipliers = [1]*72 + [1]*16 + [2]*8 + [4]*4 + [8]*2 + [8]*1 + [16]*1
        assert max_possible_score(104, multipliers) == 4000
