import pytest
from pydantic import ValidationError

from project_ted.strategy import (
    Chip,
    GameweekWindow,
    Position,
    PositionRule,
    SeasonPolicy,
    UnsupportedSeasonError,
    season_policy_for,
)


def test_returns_the_verified_2026_27_policy() -> None:
    policy = season_policy_for("2026/27")

    assert policy.season == "2026/27"
    assert policy.total_gameweeks == 38

    assert policy.squad_size == 15
    assert policy.starting_size == 11
    assert policy.max_players_per_team == 3
    assert policy.budget_tenths == 1000

    assert policy.free_transfers_per_gameweek == 1
    assert policy.maximum_free_transfers == 5
    assert policy.additional_transfer_cost_points == 4
    assert policy.maximum_chips_per_gameweek == 1


def test_defines_the_complete_squad_composition() -> None:
    policy = season_policy_for("2026/27")

    assert policy.position_rule(Position.GOALKEEPER) == PositionRule(
        position=Position.GOALKEEPER,
        squad_count=2,
        minimum_starters=1,
        maximum_starters=1,
    )
    assert policy.position_rule(Position.DEFENDER).squad_count == 5
    assert policy.position_rule(Position.MIDFIELDER).squad_count == 5
    assert policy.position_rule(Position.FORWARD).squad_count == 3
    assert {rule.position for rule in policy.positions} == set(Position)


@pytest.mark.parametrize(
    ("chip", "gameweek", "expected"),
    [
        (Chip.WILDCARD, 1, False),
        (Chip.WILDCARD, 2, True),
        (Chip.WILDCARD, 19, True),
        (Chip.WILDCARD, 20, True),
        (Chip.WILDCARD, 38, True),
        (Chip.FREE_HIT, 1, False),
        (Chip.FREE_HIT, 2, True),
        (Chip.FREE_HIT, 19, True),
        (Chip.FREE_HIT, 20, True),
        (Chip.BENCH_BOOST, 1, True),
        (Chip.BENCH_BOOST, 19, True),
        (Chip.BENCH_BOOST, 20, True),
        (Chip.TRIPLE_CAPTAIN, 1, True),
        (Chip.TRIPLE_CAPTAIN, 38, True),
    ],
)
def test_defines_chip_availability(
    chip: Chip,
    gameweek: int,
    expected: bool,
) -> None:
    policy = season_policy_for("2026/27")

    assert policy.chip_rule(chip).is_available_in(gameweek) is expected


def test_each_chip_can_be_used_once_in_each_half() -> None:
    policy = season_policy_for("2026/27")

    assert {rule.chip for rule in policy.chips} == set(Chip)

    for chip in Chip:
        rule = policy.chip_rule(chip)

        assert rule.uses_per_window == 1
        assert len(rule.availability_windows) == 2


def test_free_hit_cannot_be_used_in_consecutive_gameweeks() -> None:
    policy = season_policy_for("2026/27")

    free_hit = policy.chip_rule(Chip.FREE_HIT)

    assert free_hit.minimum_gameweek_difference_between_uses == 2


def test_rejects_an_unsupported_season() -> None:
    with pytest.raises(
        UnsupportedSeasonError,
        match="No verified FPL policy exists for season 2027/28",
    ):
        season_policy_for("2027/28")


def test_season_policy_is_immutable() -> None:
    policy = season_policy_for("2026/27")

    with pytest.raises(ValidationError, match="Instance is frozen"):
        policy.maximum_free_transfers = 4


def test_rejects_a_reversed_gameweek_window() -> None:
    with pytest.raises(
        ValidationError,
        match="first gameweek must not be after last gameweek",
    ):
        GameweekWindow(
            first_gameweek=20,
            last_gameweek=19,
        )


def test_rejects_invalid_position_starter_limits() -> None:
    with pytest.raises(
        ValidationError,
        match="maximum starters must not exceed squad count",
    ):
        PositionRule(
            position=Position.FORWARD,
            squad_count=3,
            minimum_starters=1,
            maximum_starters=4,
        )


def test_policy_rejects_duplicate_positions() -> None:
    policy = season_policy_for("2026/27")
    policy_data: dict[str, object] = policy.model_dump()
    policy_data["positions"] = (
        *policy.positions[:-1],
        policy.positions[0],
    )

    with pytest.raises(
        ValidationError,
        match="positions must not contain duplicates",
    ):
        SeasonPolicy.model_validate(policy_data)


def test_policy_rejects_duplicate_chips() -> None:
    policy = season_policy_for("2026/27")
    policy_data: dict[str, object] = policy.model_dump()
    policy_data["chips"] = (
        *policy.chips[:-1],
        policy.chips[0],
    )

    with pytest.raises(
        ValidationError,
        match="chips must not contain duplicates",
    ):
        SeasonPolicy.model_validate(policy_data)
