from dataclasses import dataclass

from project_ted.projections import ProjectionSet, build_player_projections


@dataclass(frozen=True)
class ProjectionPlayer:
    id: int
    team_id: int
    can_select: bool = True
    status: str = "a"
    chance_of_playing_next_round: int | None = None
    expected_points_next: float | None = 5.0
    form: float = 0.0
    points_per_game: float = 0.0
    minutes: int = 0


@dataclass(frozen=True)
class ProjectionFixture:
    gameweek: int | None
    home_team_id: int
    away_team_id: int
    home_difficulty: int | None
    away_difficulty: int | None


def fixture(
    gameweek: int,
    difficulty: int,
) -> ProjectionFixture:
    return ProjectionFixture(
        gameweek=gameweek,
        home_team_id=1,
        away_team_id=2,
        home_difficulty=difficulty,
        away_difficulty=3,
    )


def build(
    *,
    players: tuple[ProjectionPlayer, ...] | None = None,
    fixtures: tuple[ProjectionFixture, ...] | None = None,
    first_gameweek: int = 1,
    total_gameweeks: int = 38,
) -> ProjectionSet:
    return build_player_projections(
        season="2026/27",
        first_gameweek=first_gameweek,
        total_gameweeks=total_gameweeks,
        players=(players if players is not None else (ProjectionPlayer(id=1, team_id=1),)),
        fixtures=(
            fixtures
            if fixtures is not None
            else tuple(
                fixture(gameweek, difficulty)
                for gameweek, difficulty in enumerate(
                    (1, 2, 3, 4, 5),
                    start=1,
                )
            )
        ),
    )


def test_projects_the_next_six_gameweeks() -> None:
    projections = build()
    player = projections.players[0]

    assert projections.first_gameweek == 1
    assert projections.last_gameweek == 6
    assert [item.gameweek for item in player.gameweeks] == [1, 2, 3, 4, 5, 6]
    assert [item.projected_points_milli for item in player.gameweeks] == [
        6000,
        5500,
        5000,
        4500,
        4000,
        0,
    ]
    assert player.total_points_milli == 25000


def test_stops_the_horizon_at_the_end_of_the_season() -> None:
    projections = build(
        first_gameweek=36,
        total_gameweeks=38,
        fixtures=(
            fixture(36, 3),
            fixture(37, 3),
            fixture(38, 3),
        ),
    )

    assert projections.first_gameweek == 36
    assert projections.last_gameweek == 38
    assert [item.gameweek for item in projections.players[0].gameweeks] == [36, 37, 38]


def test_sums_double_gameweek_fixtures_and_scores_blanks_as_zero() -> None:
    projections = build(
        fixtures=(
            fixture(1, 2),
            fixture(1, 4),
        ),
    )
    gameweeks = projections.players[0].gameweeks

    assert gameweeks[0].fixture_count == 2
    assert gameweeks[0].projected_points_milli == 10000
    assert gameweeks[1].fixture_count == 0
    assert gameweeks[1].projected_points_milli == 0


def test_reduces_projection_by_current_availability() -> None:
    projections = build(
        players=(
            ProjectionPlayer(
                id=1,
                team_id=1,
                chance_of_playing_next_round=50,
            ),
        ),
        fixtures=(fixture(1, 3),),
    )
    player = projections.players[0]

    assert player.availability_percent == 50
    assert player.gameweeks[0].projected_points_milli == 2500


def test_unselectable_players_project_zero_points() -> None:
    projections = build(
        players=(
            ProjectionPlayer(
                id=1,
                team_id=1,
                can_select=False,
            ),
        ),
        fixtures=(fixture(1, 1),),
    )

    assert projections.players[0].availability_percent == 0
    assert projections.players[0].total_points_milli == 0


def test_blends_current_metrics_after_a_player_has_minutes() -> None:
    projections = build(
        players=(
            ProjectionPlayer(
                id=1,
                team_id=1,
                expected_points_next=5.0,
                form=4.0,
                points_per_game=6.0,
                minutes=900,
            ),
        ),
        fixtures=(fixture(1, 3),),
    )

    assert projections.players[0].baseline_points_milli == 4900


def test_projection_is_independent_of_input_order() -> None:
    players = (
        ProjectionPlayer(id=2, team_id=1),
        ProjectionPlayer(id=1, team_id=1),
    )
    fixtures = (
        fixture(2, 4),
        fixture(1, 2),
    )

    first = build(
        players=players,
        fixtures=fixtures,
    )
    second = build(
        players=tuple(reversed(players)),
        fixtures=tuple(reversed(fixtures)),
    )

    assert first == second
    assert [player.player_id for player in first.players] == [1, 2]
