from datetime import UTC, datetime

import pytest

from project_ted.agent.tools import (
    ComparePlayersInput,
    FindPlayersInput,
    GetFixturesInput,
    GetPlayerNewsInput,
    compare_players,
    find_players,
    get_fixtures,
    get_player_news,
)
from project_ted.data.catalog import (
    CatalogPlayer,
    Fixture,
    Gameweek,
    SnapshotCatalog,
    Team,
)
from project_ted.engine.models import Position


def make_catalog() -> SnapshotCatalog:
    teams = (
        Team(
            id=1,
            name="Manchester City",
            short_name="MCI",
            strength=5,
        ),
        Team(
            id=2,
            name="Arsenal",
            short_name="ARS",
            strength=5,
        ),
        Team(
            id=3,
            name="Chelsea",
            short_name="CHE",
            strength=4,
        ),
    )
    players = (
        CatalogPlayer(
            id=1,
            first_name="Erling",
            second_name="Haaland",
            web_name="Haaland",
            team=1,
            element_type=Position.FORWARD,
            now_cost=145,
            status="a",
            news="",
            chance_of_playing_next_round=100,
            form=8.0,
            points_per_game=7.2,
            selected_by_percent=60.0,
            total_points=220,
            minutes=2800,
        ),
        CatalogPlayer(
            id=2,
            first_name="Bukayo",
            second_name="Saka",
            web_name="Saka",
            team=2,
            element_type=Position.MIDFIELDER,
            now_cost=105,
            status="d",
            news="Knock - 75% chance of playing",
            chance_of_playing_next_round=75,
            form=7.5,
            points_per_game=6.5,
            selected_by_percent=45.0,
            total_points=200,
            minutes=2700,
        ),
        CatalogPlayer(
            id=3,
            first_name="Cole",
            second_name="Palmer",
            web_name="Palmer",
            team=3,
            element_type=Position.MIDFIELDER,
            now_cost=110,
            status="a",
            news="",
            chance_of_playing_next_round=None,
            form=6.5,
            points_per_game=6.0,
            selected_by_percent=40.0,
            total_points=190,
            minutes=2600,
        ),
    )
    gameweeks = (
        Gameweek(
            id=1,
            name="Gameweek 1",
            deadline_time=datetime(
                2026,
                8,
                21,
                17,
                30,
                tzinfo=UTC,
            ),
            finished=False,
            data_checked=False,
            is_current=False,
            is_next=True,
        ),
        Gameweek(
            id=2,
            name="Gameweek 2",
            deadline_time=datetime(
                2026,
                8,
                28,
                17,
                30,
                tzinfo=UTC,
            ),
            finished=False,
            data_checked=False,
            is_current=False,
            is_next=False,
        ),
    )
    fixtures = (
        Fixture(
            id=10,
            event=1,
            kickoff_time=datetime(
                2026,
                8,
                21,
                19,
                tzinfo=UTC,
            ),
            team_h=1,
            team_a=2,
            team_h_difficulty=3,
            team_a_difficulty=4,
            started=False,
            finished=False,
            team_h_score=None,
            team_a_score=None,
        ),
        Fixture(
            id=11,
            event=2,
            kickoff_time=datetime(
                2026,
                8,
                28,
                19,
                tzinfo=UTC,
            ),
            team_h=3,
            team_a=2,
            team_h_difficulty=3,
            team_a_difficulty=3,
            started=False,
            finished=False,
            team_h_score=None,
            team_a_score=None,
        ),
    )

    return SnapshotCatalog(
        snapshot_id="test-snapshot",
        players=players,
        teams=teams,
        gameweeks=gameweeks,
        fixtures=fixtures,
    )


def test_find_players_searches_names_without_case_sensitivity() -> None:
    results = find_players(
        make_catalog(),
        FindPlayersInput(query="sAkA"),
    )

    assert len(results) == 1
    assert results[0].player_id == 2
    assert results[0].team == "Arsenal"
    assert results[0].price_millions == 10.5


def test_find_players_filters_and_sorts_players() -> None:
    results = find_players(
        make_catalog(),
        FindPlayersInput(
            position=Position.MIDFIELDER,
            maximum_price_tenths=110,
            sort_by="form",
        ),
    )

    assert [player.player_id for player in results] == [2, 3]


def test_compare_players_preserves_the_requested_order() -> None:
    results = compare_players(
        make_catalog(),
        ComparePlayersInput(player_ids=(3, 1)),
    )

    assert [player.player_id for player in results] == [3, 1]


def test_compare_players_rejects_unknown_players() -> None:
    with pytest.raises(ValueError, match="unknown player IDs: 99"):
        compare_players(
            make_catalog(),
            ComparePlayersInput(player_ids=(1, 99)),
        )


def test_get_fixtures_filters_by_gameweek_and_team() -> None:
    results = get_fixtures(
        make_catalog(),
        GetFixturesInput(gameweek=1, team_id=2),
    )

    assert len(results) == 1
    assert results[0].home_team == "Manchester City"
    assert results[0].away_team == "Arsenal"


def test_get_player_news_returns_flagged_players() -> None:
    results = get_player_news(
        make_catalog(),
        GetPlayerNewsInput(),
    )

    assert [player.player_id for player in results] == [2]
    assert results[0].chance_of_playing_next_round == 75
