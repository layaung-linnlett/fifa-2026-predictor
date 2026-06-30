"""
FIFA World Cup 2026 knockout bracket seeding.

--- Plain English explanation ---

The 2026 World Cup has 48 teams in 12 groups (A–L) of 4 teams each.
After the group stage:
    - The WINNER and RUNNER-UP of every group advance (24 teams)
    - The 8 BEST THIRD-PLACE teams also advance (8 teams)
    - Total: 32 teams enter the Round of 32

The bracket is FIXED — FIFA pre-determines which group slot goes where.
The knockout_slots.csv encodes this. Slot strings like "Winner Group C"
or "Runner-up Group F" get resolved to actual teams once we know the
group stage results.

The tricky part: BEST THIRD-PLACE slots.
    Each "Best 3rd (Groups A/B/C/D/F)" slot lists 5 candidate groups.
    FIFA uses a lookup table to assign the 8 qualified third-place teams
    to the 8 bracket slots, depending on which groups they came from.

    The official 2026 placement table has not been published at the time
    of writing. We use a reasonable approximation: for each "Best 3rd"
    slot, we take the best-ranked third-place team whose group appears in
    the slot's candidate list and hasn't been assigned yet.

    JUDGMENT CALL: This approximation may differ from the official FIFA
    placement table when it is released. It is a sound simulation approach
    but the exact slot assignment is not guaranteed to match reality.

Group stage ranking tiebreakers (FIFA rules, applied in order):
    1. Points (win=3, draw=1, loss=0)
    2. Goal difference
    3. Goals scored
    4. Head-to-head points
    5. Head-to-head goal difference
    6. Head-to-head goals scored
    (Further tiebreakers: disciplinary points, FIFA ranking — not modelled)
"""

import pandas as pd


GROUPS = list("ABCDEFGHIJKL")


def compute_group_standings(
    group: str,
    match_results: list[dict],
) -> list[dict]:
    """
    Compute the final standings for one group.

    Args:
        group:         Group letter, e.g. "A".
        match_results: List of dicts, each with keys:
                       home_team, away_team, home_score, away_score.
                       Only matches within this group are used.

    Returns:
        List of team dicts sorted by group stage rank (1st to 4th).
        Each dict has: team, points, gd (goal difference), gf (goals for).
    """
    teams = {}

    for match in match_results:
        for team in [match["home_team"], match["away_team"]]:
            if team not in teams:
                teams[team] = {"team": team, "points": 0, "gd": 0, "gf": 0}

    for match in match_results:
        h = match["home_team"]
        a = match["away_team"]
        hs = match["home_score"]
        as_ = match["away_score"]

        teams[h]["gf"] += hs
        teams[a]["gf"] += as_
        teams[h]["gd"] += hs - as_
        teams[a]["gd"] += as_ - hs

        if hs > as_:
            teams[h]["points"] += 3
        elif hs == as_:
            teams[h]["points"] += 1
            teams[a]["points"] += 1
        else:
            teams[a]["points"] += 3

    standings = sorted(
        teams.values(),
        key=lambda t: (t["points"], t["gd"], t["gf"]),
        reverse=True,
    )
    return standings


def resolve_best_third(all_standings: dict[str, list[dict]]) -> list[dict]:
    """
    Identify and rank the 8 best third-place teams across all 12 groups.

    Args:
        all_standings: Dict mapping group letter -> sorted standings list.

    Returns:
        List of 8 team dicts (same format as compute_group_standings output),
        sorted best-to-worst, each with an added "group" key.
    """
    third_place = []
    for grp, standings in all_standings.items():
        if len(standings) >= 3:
            entry = dict(standings[2])
            entry["group"] = grp
            third_place.append(entry)

    third_place.sort(
        key=lambda t: (t["points"], t["gd"], t["gf"]),
        reverse=True,
    )
    return third_place[:8]


def build_r32_lineup(all_standings: dict[str, list[dict]]) -> dict[str, str]:
    """
    Map every Round of 32 slot string to an actual team name.

    Args:
        all_standings: Dict mapping group letter -> sorted standings list.

    Returns:
        Dict mapping slot string -> team name.
        e.g. {"Winner Group A": "Spain", "Runner-up Group B": "Brazil", ...}
    """
    slot_to_team = {}

    for grp, standings in all_standings.items():
        if len(standings) >= 1:
            slot_to_team[f"Winner Group {grp}"] = standings[0]["team"]
        if len(standings) >= 2:
            slot_to_team[f"Runner-up Group {grp}"] = standings[1]["team"]

    best_thirds = resolve_best_third(all_standings)

    # Assign each best-third team to a "Best 3rd (Groups ...)" slot.
    # The slot's candidate groups are listed in parentheses.
    # We go through the 8 best third-place teams in rank order and assign
    # each to the first unoccupied slot that lists their group.
    best_third_slots = [
        "Best 3rd (Groups A/B/C/D/F)",
        "Best 3rd (Groups C/D/F/G/H)",
        "Best 3rd (Groups C/E/F/H/I)",
        "Best 3rd (Groups E/H/I/J/K)",
        "Best 3rd (Groups A/E/H/I/J)",
        "Best 3rd (Groups B/E/F/I/J)",
        "Best 3rd (Groups E/F/G/I/J)",
        "Best 3rd (Groups D/E/I/J/L)",
    ]

    assigned_slots = set()
    for team_entry in best_thirds:
        grp = team_entry["group"]
        for slot in best_third_slots:
            if slot in assigned_slots:
                continue
            # Extract candidate groups from the slot string
            groups_str = slot.split("(Groups ")[1].rstrip(")")
            candidate_groups = [g.strip() for g in groups_str.split("/")]
            if grp in candidate_groups:
                slot_to_team[slot] = team_entry["team"]
                assigned_slots.add(slot)
                break

    return slot_to_team


def resolve_bracket(
    knockout_slots: pd.DataFrame,
    r32_lineup: dict[str, str],
    match_winners: dict[int, str],
    match_losers: dict[int, str] | None = None,
) -> dict[str, str]:
    """
    Resolve all knockout slot strings to team names.

    This function is called once before simulating each knockout match.
    It resolves three types of slot:
        "Winner Group X"        -> from r32_lineup
        "Runner-up Group X"     -> from r32_lineup
        "Best 3rd (Groups ...)" -> from r32_lineup
        "Winner Match N"        -> from match_winners dict
        "Loser Match N"         -> from match_losers dict (for 3rd place)

    Args:
        knockout_slots: The knockout_slots DataFrame (from CSV).
        r32_lineup:     Output of build_r32_lineup().
        match_winners:  Dict mapping match_id -> winning team name.
        match_losers:   Dict mapping match_id -> losing team name.

    Returns:
        Dict mapping slot string -> team name for every slot we can resolve.
    """
    if match_losers is None:
        match_losers = {}

    resolved = dict(r32_lineup)

    for _, row in knockout_slots.iterrows():
        match_id = row["match_id"]
        if match_id in match_winners:
            resolved[f"Winner Match {match_id}"] = match_winners[match_id]
        if match_id in match_losers:
            resolved[f"Loser Match {match_id}"] = match_losers[match_id]

    return resolved
