"""
Corners and cards prediction model.

--- Honest statement of limitations ---

The ideal version of this module would:
    1. Load a historical dataset of corners and cards per international match
    2. Compute each team's average corners won / cards received per game
    3. Fit a simple model: predicted_corners = avg(team_a) + avg(team_b)

We do NOT have that dataset. The results.csv file only contains goals,
not corners or cards. A corners/cards dataset for international football
is not freely available in the same clean form as goal results.

What we do instead:
    - Use empirically reasonable BASE rates drawn from World Cup averages
      (roughly 10 corners and 3 yellow cards per match at major tournaments)
    - Apply per-team STYLE MULTIPLIERS that encode qualitative knowledge
      about each team's playing style (e.g. Spain win more corners through
      possession; Uruguay receive more cards through physicality)

IMPORTANT: These multipliers are ESTIMATES, not measurements from data.
They are informed guesses, not statistical fits. The outputs should be
treated as plausible approximations, not precise predictions.

This is documented transparently in docs/methodology.md so that anyone
reading the project — including a hiring manager — understands exactly
what is modelled vs estimated.

Why keep it at all?
    The competition requires corners/cards predictions for all 104 matches.
    Honest estimated values based on qualitative knowledge are more useful
    than refusing to predict. We just label them clearly as estimates.
"""

# ---------------------------------------------------------------------------
# BASE RATES — drawn from World Cup tournament averages
#
# JUDGMENT CALL:
#   corners  = 10.0 per match  (source: FIFA World Cup match stats, ~10–11)
#   yellows  = 3.0 per match   (source: major tournament averages, ~3–4)
#   reds     = 0.15 per match  (source: rare — roughly 1 per 7 matches)
#
# These are split evenly between the two teams as the starting point,
# then adjusted by each team's style multiplier.
# ---------------------------------------------------------------------------
BASE_CORNERS = 10.0
BASE_YELLOWS = 3.0
BASE_REDS = 0.15

# ---------------------------------------------------------------------------
# STYLE MULTIPLIERS — per team, per stat
#
# Each value is relative to 1.0 (average).
#   > 1.0 means this team tends to win MORE corners / receive MORE cards
#   < 1.0 means this team tends to win FEWER corners / receive FEWER cards
#
# The combined prediction formula is:
#   corners = BASE_CORNERS * (team_a["corners"] + team_b["corners"]) / 2
#
# Dividing by 2 keeps the total around BASE_CORNERS regardless of the
# teams involved (two average teams give exactly BASE_CORNERS).
#
# Sources for the qualitative judgements:
#   - Possession-heavy teams (Spain, Brazil) win more corners
#   - Physical / defensive teams (Uruguay, Colombia) earn more cards
#   - Technical disciplined teams (Japan, Germany) earn fewer cards
#   - Smaller nations with less international experience tend toward
#     more cards as they compete against stronger opposition
# ---------------------------------------------------------------------------
STYLE: dict[str, dict[str, float]] = {
    "Argentina":      {"corners": 1.1, "yellows": 1.2, "reds": 1.1},
    "Australia":      {"corners": 0.9, "yellows": 1.0, "reds": 0.8},
    "Austria":        {"corners": 1.0, "yellows": 1.1, "reds": 0.9},
    "Belgium":        {"corners": 1.1, "yellows": 0.9, "reds": 0.7},
    "Brazil":         {"corners": 1.2, "yellows": 1.0, "reds": 0.8},
    "Cabo Verde":     {"corners": 0.8, "yellows": 1.2, "reds": 1.1},
    "Canada":         {"corners": 1.0, "yellows": 1.0, "reds": 0.9},
    "Colombia":       {"corners": 1.1, "yellows": 1.3, "reds": 1.2},
    "Croatia":        {"corners": 1.0, "yellows": 1.0, "reds": 0.8},
    "Curaçao":        {"corners": 0.8, "yellows": 1.1, "reds": 1.0},
    "Côte d'Ivoire":  {"corners": 1.0, "yellows": 1.1, "reds": 1.0},
    "Ecuador":        {"corners": 0.9, "yellows": 1.1, "reds": 1.0},
    "Egypt":          {"corners": 0.9, "yellows": 1.1, "reds": 1.0},
    "England":        {"corners": 1.1, "yellows": 0.9, "reds": 0.7},
    "France":         {"corners": 1.1, "yellows": 1.0, "reds": 0.8},
    "Germany":        {"corners": 1.0, "yellows": 0.9, "reds": 0.7},
    "Ghana":          {"corners": 0.9, "yellows": 1.1, "reds": 1.0},
    "Haiti":          {"corners": 0.8, "yellows": 1.2, "reds": 1.1},
    "Iran":           {"corners": 0.9, "yellows": 1.2, "reds": 1.0},
    "Japan":          {"corners": 1.0, "yellows": 0.8, "reds": 0.6},
    "Jordan":         {"corners": 0.8, "yellows": 1.1, "reds": 1.0},
    "Mexico":         {"corners": 1.0, "yellows": 1.1, "reds": 1.0},
    "Morocco":        {"corners": 0.9, "yellows": 1.1, "reds": 0.9},
    "Netherlands":    {"corners": 1.1, "yellows": 1.0, "reds": 0.8},
    "New Zealand":    {"corners": 0.8, "yellows": 0.9, "reds": 0.8},
    "Norway":         {"corners": 1.0, "yellows": 1.0, "reds": 0.8},
    "Panama":         {"corners": 0.8, "yellows": 1.2, "reds": 1.1},
    "Paraguay":       {"corners": 0.9, "yellows": 1.2, "reds": 1.1},
    "Portugal":       {"corners": 1.1, "yellows": 1.1, "reds": 0.9},
    "Qatar":          {"corners": 0.9, "yellows": 1.0, "reds": 0.9},
    "Algeria":        {"corners": 0.9, "yellows": 1.1, "reds": 1.0},
    "Saudi Arabia":   {"corners": 0.9, "yellows": 1.0, "reds": 0.9},
    "Scotland":       {"corners": 1.0, "yellows": 1.0, "reds": 0.8},
    "Senegal":        {"corners": 0.9, "yellows": 1.1, "reds": 1.0},
    "South Africa":   {"corners": 0.9, "yellows": 1.1, "reds": 1.0},
    "South Korea":    {"corners": 1.0, "yellows": 1.0, "reds": 0.8},
    "Spain":          {"corners": 1.2, "yellows": 0.9, "reds": 0.7},
    "Switzerland":    {"corners": 1.0, "yellows": 0.9, "reds": 0.7},
    "Tunisia":        {"corners": 0.9, "yellows": 1.2, "reds": 1.0},
    "Uruguay":        {"corners": 1.0, "yellows": 1.3, "reds": 1.2},
    "USA":            {"corners": 1.0, "yellows": 0.9, "reds": 0.8},
    "Uzbekistan":     {"corners": 0.9, "yellows": 1.1, "reds": 1.0},
    # Playoff slots — unknown teams, use neutral averages
    "UEFA Playoff A": {"corners": 1.0, "yellows": 1.0, "reds": 1.0},
    "UEFA Playoff B": {"corners": 1.0, "yellows": 1.0, "reds": 1.0},
    "UEFA Playoff C": {"corners": 1.0, "yellows": 1.0, "reds": 1.0},
    "UEFA Playoff D": {"corners": 1.0, "yellows": 1.0, "reds": 1.0},
    "FIFA Playoff 1": {"corners": 1.0, "yellows": 1.0, "reds": 1.0},
    "FIFA Playoff 2": {"corners": 1.0, "yellows": 1.0, "reds": 1.0},
}

# Neutral fallback for any team not in STYLE (e.g. future playoff winners)
_NEUTRAL = {"corners": 1.0, "yellows": 1.0, "reds": 1.0}


def predict_corners_cards(
    team_a: str,
    team_b: str,
    base_corners: float = BASE_CORNERS,
    base_yellows: float = BASE_YELLOWS,
    base_reds: float = BASE_REDS,
) -> tuple:
    """
    Predict corners, yellow cards, and red cards for a match.

    Returns (corners, yellows, reds) as floats rounded to 1 decimal place.

    NOTE: These are estimated values, not data-fitted predictions.
    See module docstring for full explanation of limitations.

    The formula keeps totals near the base rate for two average teams:
        corners = base_corners * (style_a + style_b) / 2
    """
    style_a = STYLE.get(team_a, _NEUTRAL)
    style_b = STYLE.get(team_b, _NEUTRAL)

    corners = base_corners * (style_a["corners"] + style_b["corners"]) / 2
    yellows = base_yellows * (style_a["yellows"] + style_b["yellows"]) / 2
    reds    = base_reds    * (style_a["reds"]    + style_b["reds"])    / 2

    return round(corners, 1), round(yellows, 1), round(reds, 1)


def known_teams() -> set:
    """Return the set of teams we have style data for."""
    return set(STYLE.keys())
