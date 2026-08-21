from datetime import UTC, datetime
from uuid import UUID

import pytest
from project_ted.gameweek_scoring import (
    GameweekScoringError,
    OfficialGameweekPoints,
    OfficialPlayerPoints,
    evaluate_gameweek,
)
from pydantic import ValidationError

from project_ted.decision import LockedGameweekDecision
from project_ted.planning import AgentProvider, GameweekPlan
from project_ted.strategy import Chip, Position, season_policy_for

POSITIONS = {
    1: Position.GOALKEEPER,
    2: Position.DEFENDER,
    3: Position.DEFENDER,
    4: Position.DEFENDER,
    5: Position.MIDFIELDER,
    6: Position.MIDFIELDER,
    7: Position.MIDFIELDER,
    8: Position.MIDFIELDER,
    9: Position.FORWARD,
    10: Position.FORWARD,
    11: Position.FORWARD,
    12: Position.GOALKEEPER,
    13: Position.DEFENDER,
    14: Position.MIDFIELDER,
    15: Position.DEFENDER,
}


def decision(
    *,
    chip: Chip | None = None,
    transfer_cost_points: int = 4,
) -> LockedGameweekDecision:
    return LockedGameweekDecision(
        source_run_id=UUID("b180ace4-911f-48ab-b050-b6b286dd3949"),
        provider=AgentProvider.OPENAI,
        model_name="gpt-test",
        plan=GameweekPlan(
            season="2026/27",
            gameweek=1,
            squad=tuple(range(1, 16)),
            starting_xi=tuple(range(1, 12)),
            bench=(14, 13, 15, 12),
            captain_id=9,
            vice_captain_id=5,
            chip=chip,
            rationale="A test decision.",
        ),
        transfer_cost_points=transfer_cost_points,
        deadline_at=datetime(2026, 8, 21, 17, 30, tzinfo=UTC),
        locked_at=datetime(2026, 8, 20, 12, tzinfo=UTC),
    )


def official_points(
    *,
    did_not_play: tuple[int, ...] = (),
    omitted_player_ids: tuple[int, ...] = (),
    season: str = "2026/27",
    gameweek: int = 1,
) -> OfficialGameweekPoints:
    return OfficialGameweekPoints(
        season=season,
        gameweek=gameweek,
        finalized_at=datetime(2026, 8, 24, 22, tzinfo=UTC),
        players=tuple(
            OfficialPlayerPoints(
                player_id=player_id,
                position=POSITIONS[player_id],
                points=player_id,
                played=player_id not in did_not_play,
            )
            for player_id in range(1, 16)
            if player_id not in omitted_player_ids
        ),
    )


def test_scores_the_starting_team_captain_and_transfer_cost() -> None:
    result = evaluate_gameweek(
        decision(),
        official_points(),
        season_policy_for("2026/27"),
    )

    assert result.gross_points == 75
    assert result.transfer_cost_points == 4
    assert result.total_points == 71
    assert result.effective_captain_id == 9
    assert result.substitutions == ()


def test_uses_the_first_formation_eligible_outfield_substitute() -> None:
    result = evaluate_gameweek(
        decision(),
        official_points(did_not_play=(2,)),
        season_policy_for("2026/27"),
    )

    assert result.substitutions[0].player_out_id == 2
    assert result.substitutions[0].player_in_id == 13
    assert result.gross_points == 86


def test_uses_the_highest_priority_eligible_substitute() -> None:
    result = evaluate_gameweek(
        decision(),
        official_points(did_not_play=(5,)),
        season_policy_for("2026/27"),
    )

    assert result.substitutions[0].player_out_id == 5
    assert result.substitutions[0].player_in_id == 14


def test_only_the_bench_goalkeeper_can_replace_the_goalkeeper() -> None:
    result = evaluate_gameweek(
        decision(),
        official_points(did_not_play=(1,)),
        season_policy_for("2026/27"),
    )

    assert result.substitutions[0].player_out_id == 1
    assert result.substitutions[0].player_in_id == 12


def test_vice_captain_takes_over_when_captain_does_not_play() -> None:
    result = evaluate_gameweek(
        decision(),
        official_points(did_not_play=(9,)),
        season_policy_for("2026/27"),
    )

    assert result.effective_captain_id == 5
    assert result.gross_points == 76


def test_no_captain_bonus_when_captain_and_vice_do_not_play() -> None:
    result = evaluate_gameweek(
        decision(),
        official_points(did_not_play=(5, 9)),
        season_policy_for("2026/27"),
    )

    assert result.effective_captain_id is None


def test_triple_captain_multiplier_moves_to_the_vice_captain() -> None:
    result = evaluate_gameweek(
        decision(chip=Chip.TRIPLE_CAPTAIN),
        official_points(did_not_play=(9,)),
        season_policy_for("2026/27"),
    )

    assert result.effective_captain_id == 5
    assert result.gross_points == 81


def test_bench_boost_counts_the_entire_squad() -> None:
    result = evaluate_gameweek(
        decision(chip=Chip.BENCH_BOOST),
        official_points(),
        season_policy_for("2026/27"),
    )

    assert result.gross_points == 129
    assert result.substitutions == ()


def test_rejects_missing_official_player_points() -> None:
    with pytest.raises(
        GameweekScoringError,
        match="missing official points for player IDs: 15",
    ):
        evaluate_gameweek(
            decision(),
            official_points(omitted_player_ids=(15,)),
            season_policy_for("2026/27"),
        )


def test_rejects_points_for_another_gameweek() -> None:
    with pytest.raises(
        GameweekScoringError,
        match="official points do not match the locked decision",
    ):
        evaluate_gameweek(
            decision(),
            official_points(gameweek=2),
            season_policy_for("2026/27"),
        )


def test_rejects_transfer_cost_that_does_not_match_policy() -> None:
    with pytest.raises(
        GameweekScoringError,
        match="transfer cost must be a multiple of 4",
    ):
        evaluate_gameweek(
            decision(transfer_cost_points=3),
            official_points(),
            season_policy_for("2026/27"),
        )


def test_official_points_require_unique_player_ids() -> None:
    player = OfficialPlayerPoints(
        player_id=1,
        position=Position.GOALKEEPER,
        points=2,
        played=True,
    )

    with pytest.raises(
        ValidationError,
        match="official player IDs must be unique",
    ):
        OfficialGameweekPoints(
            season="2026/27",
            gameweek=1,
            finalized_at=datetime(2026, 8, 24, 22, tzinfo=UTC),
            players=(player, player),
        )
