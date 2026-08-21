from datetime import UTC, datetime

import httpx
import pytest
import respx

from project_ted.fpl import (
    FplDataError,
    Gameweek,
    InvalidPlanError,
    PlanningContext,
    Player,
    Team,
    fetch_planning_context,
)
from project_ted.planning import GameweekPlan
from project_ted.strategy import Position, season_policy_for

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"


def bootstrap_payload() -> dict[str, object]:
    return {
        "events": [
            {
                "id": 1,
                "name": "Gameweek 1",
                "deadline_time": "2026-08-21T17:30:00Z",
                "is_next": True,
            }
        ],
        "teams": [
            {
                "id": 1,
                "name": "Arsenal",
                "short_name": "ARS",
            },
            {
                "id": 7,
                "name": "Coventry City",
                "short_name": "COV",
            },
        ],
        "element_types": [
            {
                "id": 1,
                "singular_name_short": "GKP",
                "squad_select": 2,
                "squad_min_play": 1,
                "squad_max_play": 1,
            },
            {
                "id": 2,
                "singular_name_short": "DEF",
                "squad_select": 5,
                "squad_min_play": 3,
                "squad_max_play": 5,
            },
            {
                "id": 3,
                "singular_name_short": "MID",
                "squad_select": 5,
                "squad_min_play": 2,
                "squad_max_play": 5,
            },
            {
                "id": 4,
                "singular_name_short": "FWD",
                "squad_select": 3,
                "squad_min_play": 1,
                "squad_max_play": 3,
            },
        ],
        "game_settings": {
            "squad_squadsize": 15,
            "squad_squadplay": 11,
            "squad_team_limit": 3,
            "squad_total_spend": 1000,
        },
        "game_config": {
            "settings": {
                "static_content_url": (
                    "https://fantasy.premierleague.com/"
                    "gcs/plfpl-prod-static-content/"
                    "plfpl-production/2026_27/"
                )
            }
        },
        "elements": [
            {
                "id": 1,
                "web_name": "Raya",
                "team": 1,
                "element_type": 1,
                "now_cost": 60,
                "status": "a",
                "chance_of_playing_next_round": None,
                "news": "",
                "can_select": True,
                "total_points": 162,
                "minutes": 3330,
                "starts": 37,
                "form": "0.0",
                "points_per_game": "4.4",
                "selected_by_percent": "33.3",
                "ep_next": "4.5",
                "expected_goals": "0.00",
                "expected_assists": "0.00",
                "expected_goal_involvements": "0.00",
                "transfers_in_event": 100,
                "transfers_out_event": 20,
            }
        ],
    }


def fixtures_payload() -> list[dict[str, object]]:
    return [
        {
            "id": 1,
            "event": 1,
            "kickoff_time": "2026-08-21T19:00:00Z",
            "team_h": 1,
            "team_a": 7,
            "team_h_difficulty": 2,
            "team_a_difficulty": 5,
            "started": False,
            "finished": False,
        }
    ]


def test_fetches_one_normalized_planning_context() -> None:
    with respx.mock:
        bootstrap_route = respx.get(BOOTSTRAP_URL).mock(
            return_value=httpx.Response(
                200,
                json=bootstrap_payload(),
            )
        )
        fixtures_route = respx.get(FIXTURES_URL).mock(
            return_value=httpx.Response(
                200,
                json=fixtures_payload(),
            )
        )

        context = fetch_planning_context()

    assert bootstrap_route.called
    assert fixtures_route.called

    assert context.season == "2026/27"
    assert context.target_gameweek.id == 1
    assert context.target_gameweek.deadline_at == datetime(
        2026,
        8,
        21,
        17,
        30,
        tzinfo=UTC,
    )

    assert context.rules.squad_size == 15
    assert context.rules.starting_size == 11
    assert context.rules.max_players_per_team == 3
    assert context.rules.budget_tenths == 1000

    goalkeeper_rule = next(
        rule for rule in context.rules.positions if rule.position is Position.GOALKEEPER
    )
    assert goalkeeper_rule.squad_count == 2
    assert goalkeeper_rule.minimum_starters == 1
    assert goalkeeper_rule.maximum_starters == 1

    assert context.teams[0].name == "Arsenal"

    player = context.players[0]
    assert player.name == "Raya"
    assert player.team_id == 1
    assert player.position is Position.GOALKEEPER
    assert player.price_tenths == 60
    assert player.selected_by_percent == 33.3
    assert player.expected_points_next == 4.5

    fixture = context.fixtures[0]
    assert fixture.gameweek == 1
    assert fixture.home_team_id == 1
    assert fixture.away_team_id == 7
    assert fixture.home_difficulty == 2
    assert fixture.away_difficulty == 5


def test_rejects_bootstrap_rules_that_disagree_with_policy() -> None:
    payload = bootstrap_payload()
    game_settings = payload["game_settings"]

    assert isinstance(game_settings, dict)

    game_settings["squad_total_spend"] = 999

    with respx.mock:
        respx.get(BOOTSTRAP_URL).mock(
            return_value=httpx.Response(
                200,
                json=payload,
            )
        )
        respx.get(FIXTURES_URL).mock(
            return_value=httpx.Response(
                200,
                json=fixtures_payload(),
            )
        )

        with pytest.raises(
            FplDataError,
            match="Could not load current FPL data",
        ):
            fetch_planning_context()


def test_hides_http_failures_behind_one_fpl_error() -> None:
    with respx.mock:
        respx.get(BOOTSTRAP_URL).mock(return_value=httpx.Response(503))

        with pytest.raises(
            FplDataError,
            match="Could not load current FPL data",
        ):
            fetch_planning_context()


def validation_players() -> tuple[Player, ...]:
    positions = (
        Position.GOALKEEPER,
        Position.GOALKEEPER,
        Position.DEFENDER,
        Position.DEFENDER,
        Position.DEFENDER,
        Position.DEFENDER,
        Position.DEFENDER,
        Position.MIDFIELDER,
        Position.MIDFIELDER,
        Position.MIDFIELDER,
        Position.MIDFIELDER,
        Position.MIDFIELDER,
        Position.FORWARD,
        Position.FORWARD,
        Position.FORWARD,
    )

    return tuple(
        Player(
            id=player_id,
            name=f"Player {player_id}",
            team_id=((player_id - 1) % 5) + 1,
            position=position,
            price_tenths=50,
            status="a",
            chance_of_playing_next_round=None,
            news="",
            can_select=True,
            total_points=0,
            minutes=0,
            starts=0,
            form=0.0,
            points_per_game=0.0,
            selected_by_percent=0.0,
            expected_points_next=None,
            expected_goals=0.0,
            expected_assists=0.0,
            expected_goal_involvements=0.0,
            transfers_in_event=0,
            transfers_out_event=0,
        )
        for player_id, position in enumerate(
            positions,
            start=1,
        )
    )


def validation_context(
    *,
    players: tuple[Player, ...] | None = None,
) -> PlanningContext:
    return PlanningContext(
        fetched_at=datetime(2026, 8, 17, 12, tzinfo=UTC),
        season="2026/27",
        target_gameweek=Gameweek(
            id=1,
            name="Gameweek 1",
            deadline_at=datetime(
                2026,
                8,
                21,
                17,
                30,
                tzinfo=UTC,
            ),
        ),
        rules=season_policy_for("2026/27"),
        teams=tuple(
            Team(
                id=team_id,
                name=f"Team {team_id}",
                short_name=f"T{team_id}",
            )
            for team_id in range(1, 6)
        ),
        players=(validation_players() if players is None else players),
        fixtures=(),
    )


def valid_live_plan_data() -> dict[str, object]:
    return {
        "season": "2026/27",
        "gameweek": 1,
        "squad": tuple(range(1, 16)),
        "starting_xi": (
            1,
            3,
            4,
            5,
            6,
            8,
            9,
            10,
            11,
            13,
            14,
        ),
        "bench": (2, 7, 12, 15),
        "captain_id": 8,
        "vice_captain_id": 13,
        "rationale": "A valid test squad.",
        "risks": (),
    }


def live_plan_with(**changes: object) -> GameweekPlan:
    data = valid_live_plan_data()
    data.update(changes)
    return GameweekPlan.model_validate(data)


def test_context_accepts_a_valid_live_plan() -> None:
    plan = live_plan_with()

    assert validation_context().validate_plan(plan) is plan


def test_context_uses_verified_policy_squad_sizes() -> None:
    plan = live_plan_with(
        squad=tuple(range(1, 15)),
        bench=(2, 7, 12),
    )

    with pytest.raises(InvalidPlanError) as caught:
        validation_context().validate_plan(plan)

    assert "squad must contain 15 players; received 14" in caught.value.violations
    assert "bench must contain 4 players; received 3" in caught.value.violations


def test_context_rejects_unknown_players() -> None:
    plan = live_plan_with(
        squad=(*range(1, 15), 999),
        bench=(2, 7, 12, 999),
    )

    with pytest.raises(InvalidPlanError) as caught:
        validation_context().validate_plan(plan)

    assert "unknown player IDs: 999" in caught.value.violations


def test_context_rejects_unselectable_players() -> None:
    players = list(validation_players())
    players[7] = players[7].model_copy(update={"can_select": False})

    with pytest.raises(InvalidPlanError) as caught:
        validation_context(players=tuple(players)).validate_plan(live_plan_with())

    assert "unselectable player IDs: 8" in caught.value.violations


def test_context_enforces_budget() -> None:
    players = list(validation_players())
    players[0] = players[0].model_copy(update={"price_tenths": 400})

    with pytest.raises(InvalidPlanError) as caught:
        validation_context(players=tuple(players)).validate_plan(live_plan_with())

    assert "squad costs 1100 but budget is 1000" in caught.value.violations


def test_context_enforces_the_team_limit() -> None:
    players = list(validation_players())
    players[1] = players[1].model_copy(update={"team_id": 1})

    with pytest.raises(InvalidPlanError) as caught:
        validation_context(players=tuple(players)).validate_plan(live_plan_with())

    assert "Team 1 has 4 players; maximum is 3" in caught.value.violations


def test_context_enforces_squad_positions() -> None:
    players = list(validation_players())
    players[1] = players[1].model_copy(update={"position": Position.DEFENDER})

    with pytest.raises(InvalidPlanError) as caught:
        validation_context(players=tuple(players)).validate_plan(live_plan_with())

    assert "squad must contain 2 GKP players; received 1" in caught.value.violations


def test_context_enforces_starting_formation() -> None:
    plan = live_plan_with(
        starting_xi=(
            1,
            3,
            4,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        ),
        bench=(2, 5, 6, 7),
    )

    with pytest.raises(InvalidPlanError) as caught:
        validation_context().validate_plan(plan)

    assert (
        "starting XI must contain between 3 and 5 "
        "DEF players; received 2" in caught.value.violations
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        (
            "season",
            "2025/26",
            "plan season 2025/26 does not match context season 2026/27",
        ),
        (
            "gameweek",
            2,
            "plan gameweek 2 does not match target gameweek 1",
        ),
    ],
)
def test_context_targets_the_current_deadline(
    field: str,
    value: object,
    expected_message: str,
) -> None:
    plan = live_plan_with(**{field: value})

    with pytest.raises(InvalidPlanError) as caught:
        validation_context().validate_plan(plan)

    assert expected_message in caught.value.violations
