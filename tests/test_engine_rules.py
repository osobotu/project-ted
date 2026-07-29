from dataclasses import replace

from project_ted.engine.models import Pick, Player, Position, Squad
from project_ted.engine.rules import (
    SEASON_2026_27_RULES,
    validate_initial_squad,
)


def make_players() -> dict[int, Player]:
    """Create a legal 15-player catalogue for validation tests."""

    specifications = (
        # player_id, team_id, position, now_cost
        (1, 1, Position.GOALKEEPER, 50),
        (2, 2, Position.GOALKEEPER, 45),
        (3, 1, Position.DEFENDER, 45),
        (4, 2, Position.DEFENDER, 45),
        (5, 3, Position.DEFENDER, 45),
        (6, 4, Position.DEFENDER, 45),
        (7, 5, Position.DEFENDER, 45),
        (8, 1, Position.MIDFIELDER, 70),
        (9, 2, Position.MIDFIELDER, 70),
        (10, 3, Position.MIDFIELDER, 70),
        (11, 4, Position.MIDFIELDER, 70),
        (12, 5, Position.MIDFIELDER, 70),
        (13, 3, Position.FORWARD, 80),
        (14, 4, Position.FORWARD, 80),
        (15, 5, Position.FORWARD, 80),
    )

    return {
        player_id: Player(
            player_id=player_id,
            team_id=team_id,
            position=position,
            now_cost=now_cost,
        )
        for player_id, team_id, position, now_cost in specifications
    }


def make_valid_squad(players: dict[int, Player]) -> Squad:
    """Create a legal 1-3-4-3 lineup with four substitutes."""

    player_ids_by_squad_position = (
        1,  # Starting goalkeeper
        3,
        4,
        5,  # Three starting defenders
        8,
        9,
        10,
        11,  # Four starting midfielders
        13,
        14,
        15,  # Three starting forwards
        2,  # Substitute goalkeeper
        6,
        7,
        12,
    )

    picks = tuple(
        Pick(
            player_id=player_id,
            purchase_price=players[player_id].now_cost,
            squad_position=squad_position,
            is_captain=player_id == 13,
            is_vice_captain=player_id == 8,
        )
        for squad_position, player_id in enumerate(
            player_ids_by_squad_position,
            start=1,
        )
    )

    return Squad(
        picks=picks,
        bank=90,
    )


def test_validate_initial_squad_accepts_a_legal_squad() -> None:
    players = make_players()
    squad = make_valid_squad(players)

    errors = validate_initial_squad(
        squad,
        players,
        SEASON_2026_27_RULES,
    )

    assert errors == ()


def test_validate_initial_squad_reports_all_relevant_errors() -> None:
    players = make_players()
    players[15] = players[15].model_copy(update={"team_id": 1})
    squad = make_valid_squad(players)

    changed_picks: list[Pick] = []

    for pick in squad.picks:
        updates: dict[str, int | bool] = {}

        if pick.player_id == 5:
            updates["squad_position"] = 15
        elif pick.player_id == 12:
            updates["squad_position"] = 4

        if pick.player_id == 13:
            updates["is_captain"] = False
        elif pick.player_id == 2:
            updates["is_captain"] = True

        changed_picks.append(pick.model_copy(update=updates))

    invalid_squad = squad.model_copy(
        update={
            "picks": tuple(changed_picks),
            "bank": 91,
        }
    )

    errors = validate_initial_squad(
        invalid_squad,
        players,
        SEASON_2026_27_RULES,
    )

    assert "initial funds must total 1000, got 1001" in errors
    assert "team 1 has 4 players; maximum is 3" in errors
    assert "lineup requires 3-5 defenders, got 2" in errors
    assert "captain must be in the starting lineup" in errors


def test_validation_uses_the_supplied_rules_instead_of_a_hardcoded_limit() -> None:
    players = make_players()
    players[15] = players[15].model_copy(update={"team_id": 1})
    squad = make_valid_squad(players)

    relaxed_rules = replace(
        SEASON_2026_27_RULES,
        max_players_per_team=4,
    )

    errors = validate_initial_squad(
        squad,
        players,
        relaxed_rules,
    )

    assert errors == ()
