from dataclasses import dataclass, replace
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from project_ted.strategy import (
    Chip,
    Position,
    season_policy_for,
)
from project_ted.team_state import (
    ChipUsage,
    InvalidTeamStateError,
    OwnedPlayer,
    TeamState,
    calculate_selling_price,
    validate_team_state,
)


@dataclass(frozen=True)
class CatalogPlayer:
    id: int
    team_id: int
    position: Position
    price_tenths: int


def catalog_players() -> tuple[CatalogPlayer, ...]:
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
        CatalogPlayer(
            id=player_id,
            team_id=((player_id - 1) % 5) + 1,
            position=position,
            price_tenths=50,
        )
        for player_id, position in enumerate(
            positions,
            start=1,
        )
    )


def valid_state_data() -> dict[str, object]:
    return {
        "season": "2026/27",
        "planning_gameweek": 10,
        "squad": tuple(
            OwnedPlayer(
                player_id=player_id,
                purchase_price_tenths=50,
                selling_price_tenths=50,
            )
            for player_id in range(1, 16)
        ),
        "bank_tenths": 10,
        "free_transfers": 2,
        "used_chips": (),
        "version": 1,
        "confirmed_at": datetime(
            2026,
            10,
            20,
            12,
            tzinfo=UTC,
        ),
    }


def valid_state(**changes: object) -> TeamState:
    data = valid_state_data()
    data.update(changes)
    return TeamState.model_validate(data)


def test_accepts_a_valid_confirmed_team_state() -> None:
    state = valid_state()

    result = validate_team_state(
        state,
        season_policy_for("2026/27"),
        catalog_players(),
    )

    assert result is state
    assert state.player_ids == tuple(range(1, 16))
    assert state.squad_selling_value_tenths == 750
    assert state.team_value_tenths == 760


def test_team_state_is_immutable() -> None:
    state = valid_state()

    with pytest.raises(ValidationError, match="Instance is frozen"):
        state.bank_tenths = 20


def test_rejects_duplicate_squad_players() -> None:
    data = valid_state_data()
    squad = list(valid_state().squad)
    squad[-1] = squad[0]
    data["squad"] = tuple(squad)

    with pytest.raises(
        ValidationError,
        match="team-state player IDs must be unique",
    ):
        TeamState.model_validate(data)


def test_requires_timezone_aware_confirmation_time() -> None:
    data = valid_state_data()
    data["confirmed_at"] = datetime(2026, 10, 20, 12)

    with pytest.raises(
        ValidationError,
        match="confirmation time must include a timezone",
    ):
        TeamState.model_validate(data)


@pytest.mark.parametrize(
    (
        "purchase_price",
        "current_price",
        "expected_selling_price",
    ),
    [
        (75, 78, 76),
        (75, 77, 76),
        (75, 76, 75),
        (75, 75, 75),
        (75, 73, 73),
    ],
)
def test_calculates_fpl_selling_price(
    purchase_price: int,
    current_price: int,
    expected_selling_price: int,
) -> None:
    assert (
        calculate_selling_price(
            purchase_price,
            current_price,
        )
        == expected_selling_price
    )


def test_rejects_the_wrong_squad_size() -> None:
    state = valid_state(squad=tuple(valid_state().squad[:-1]))

    with pytest.raises(InvalidTeamStateError) as caught:
        validate_team_state(
            state,
            season_policy_for("2026/27"),
            catalog_players(),
        )

    assert "squad must contain 15 players; received 14" in caught.value.violations


def test_rejects_unknown_players() -> None:
    squad = list(valid_state().squad)
    squad[-1] = OwnedPlayer(
        player_id=999,
        purchase_price_tenths=50,
        selling_price_tenths=50,
    )
    state = valid_state(squad=tuple(squad))

    with pytest.raises(InvalidTeamStateError) as caught:
        validate_team_state(
            state,
            season_policy_for("2026/27"),
            catalog_players(),
        )

    assert "unknown player IDs: 999" in caught.value.violations


def test_enforces_position_counts() -> None:
    players = list(catalog_players())
    players[1] = replace(
        players[1],
        position=Position.DEFENDER,
    )

    with pytest.raises(InvalidTeamStateError) as caught:
        validate_team_state(
            valid_state(),
            season_policy_for("2026/27"),
            players,
        )

    assert "squad must contain 2 GKP players; received 1" in caught.value.violations


def test_enforces_the_maximum_players_per_team() -> None:
    players = list(catalog_players())
    players[1] = replace(
        players[1],
        team_id=1,
    )

    with pytest.raises(InvalidTeamStateError) as caught:
        validate_team_state(
            valid_state(),
            season_policy_for("2026/27"),
            players,
        )

    assert "team 1 has 4 players; maximum is 3" in caught.value.violations


def test_rejects_too_many_free_transfers() -> None:
    with pytest.raises(InvalidTeamStateError) as caught:
        validate_team_state(
            valid_state(free_transfers=6),
            season_policy_for("2026/27"),
            catalog_players(),
        )

    assert "free transfers must not exceed 5" in caught.value.violations


def test_verifies_each_owned_players_selling_price() -> None:
    players = list(catalog_players())
    players[0] = replace(
        players[0],
        price_tenths=52,
    )

    with pytest.raises(InvalidTeamStateError) as caught:
        validate_team_state(
            valid_state(),
            season_policy_for("2026/27"),
            players,
        )

    assert "player 1 selling price must be 51; received 50" in caught.value.violations


def test_only_one_chip_can_be_recorded_per_gameweek() -> None:
    data = valid_state_data()
    data["used_chips"] = (
        ChipUsage(
            chip=Chip.BENCH_BOOST,
            gameweek=5,
        ),
        ChipUsage(
            chip=Chip.TRIPLE_CAPTAIN,
            gameweek=5,
        ),
    )

    with pytest.raises(
        ValidationError,
        match="only one chip can be recorded for a gameweek",
    ):
        TeamState.model_validate(data)


def test_chip_usage_must_precede_the_planning_gameweek() -> None:
    data = valid_state_data()
    data["used_chips"] = (
        ChipUsage(
            chip=Chip.TRIPLE_CAPTAIN,
            gameweek=10,
        ),
    )

    with pytest.raises(
        ValidationError,
        match="chip usage must precede the planning gameweek",
    ):
        TeamState.model_validate(data)


def test_rejects_a_chip_outside_its_availability_window() -> None:
    state = valid_state(
        used_chips=(
            ChipUsage(
                chip=Chip.FREE_HIT,
                gameweek=1,
            ),
        )
    )

    with pytest.raises(InvalidTeamStateError) as caught:
        validate_team_state(
            state,
            season_policy_for("2026/27"),
            catalog_players(),
        )

    assert "free_hit is not available in gameweek 1" in caught.value.violations


def test_rejects_multiple_uses_in_one_chip_window() -> None:
    state = valid_state(
        used_chips=(
            ChipUsage(
                chip=Chip.BENCH_BOOST,
                gameweek=5,
            ),
            ChipUsage(
                chip=Chip.BENCH_BOOST,
                gameweek=8,
            ),
        )
    )

    with pytest.raises(InvalidTeamStateError) as caught:
        validate_team_state(
            state,
            season_policy_for("2026/27"),
            catalog_players(),
        )

    assert "bench_boost can be used only 1 time per window" in caught.value.violations


def test_rejects_consecutive_free_hits_across_chip_windows() -> None:
    state = valid_state(
        planning_gameweek=21,
        used_chips=(
            ChipUsage(
                chip=Chip.FREE_HIT,
                gameweek=19,
            ),
            ChipUsage(
                chip=Chip.FREE_HIT,
                gameweek=20,
            ),
        ),
    )

    with pytest.raises(InvalidTeamStateError) as caught:
        validate_team_state(
            state,
            season_policy_for("2026/27"),
            catalog_players(),
        )

    assert "free_hit uses must be at least 2 gameweeks apart" in caught.value.violations
