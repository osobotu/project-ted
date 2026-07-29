"""Season-driven FPL validation rules."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from project_ted.engine.models import Player, Position, Squad


@dataclass(frozen=True, slots=True)
class PositionRule:
    """Squad and lineup limits for one player position."""

    position: Position
    squad_count: int
    lineup_minimum: int
    lineup_maximum: int


@dataclass(frozen=True, slots=True)
class SeasonRules:
    """Immutable rules that control one FPL season."""

    season: str
    initial_budget_tenths: int
    squad_size: int
    lineup_size: int
    max_players_per_team: int
    position_rules: tuple[PositionRule, ...]


SEASON_2026_27_RULES = SeasonRules(
    season="2026/27",
    initial_budget_tenths=1000,
    squad_size=15,
    lineup_size=11,
    max_players_per_team=3,
    position_rules=(
        PositionRule(
            position=Position.GOALKEEPER,
            squad_count=2,
            lineup_minimum=1,
            lineup_maximum=1,
        ),
        PositionRule(
            position=Position.DEFENDER,
            squad_count=5,
            lineup_minimum=3,
            lineup_maximum=5,
        ),
        PositionRule(
            position=Position.MIDFIELDER,
            squad_count=5,
            lineup_minimum=2,
            lineup_maximum=5,
        ),
        PositionRule(
            position=Position.FORWARD,
            squad_count=3,
            lineup_minimum=1,
            lineup_maximum=3,
        ),
    ),
)


def validate_initial_squad(
    squad: Squad,
    players: Mapping[int, Player],
    rules: SeasonRules,
) -> tuple[str, ...]:
    """Return every rule violation found in a proposed initial squad."""

    errors: list[str] = []
    picks = squad.picks

    if len(picks) != rules.squad_size:
        errors.append(f"squad requires {rules.squad_size} players, got {len(picks)}")

    player_ids = [pick.player_id for pick in picks]
    if len(set(player_ids)) != len(player_ids):
        errors.append("squad contains duplicate players")

    squad_positions = [pick.squad_position for pick in picks]
    expected_positions = set(range(1, rules.squad_size + 1))
    if (
        len(set(squad_positions)) != len(squad_positions)
        or set(squad_positions) != expected_positions
    ):
        errors.append(f"squad positions must be exactly 1-{rules.squad_size}")

    selected_players: list[Player] = []

    for pick in picks:
        player = players.get(pick.player_id)

        if player is None:
            errors.append(f"unknown player ID: {pick.player_id}")
            continue

        selected_players.append(player)

        if pick.purchase_price != player.now_cost:
            errors.append(
                f"player {pick.player_id} purchase price must equal current price {player.now_cost}"
            )

    total_funds_tenths = sum(pick.purchase_price for pick in picks) + squad.bank
    if total_funds_tenths != rules.initial_budget_tenths:
        errors.append(
            f"initial funds must total {rules.initial_budget_tenths}, got {total_funds_tenths}"
        )

    squad_position_counts: Counter[Position] = Counter(
        player.position for player in selected_players
    )

    for position_rule in rules.position_rules:
        actual_count = squad_position_counts[position_rule.position]

        if actual_count != position_rule.squad_count:
            errors.append(
                f"squad requires {position_rule.squad_count} "
                f"{position_rule.position.value}s, got {actual_count}"
            )

    team_counts: Counter[int] = Counter(player.team_id for player in selected_players)

    for team_id, player_count in sorted(team_counts.items()):
        if player_count > rules.max_players_per_team:
            errors.append(
                f"team {team_id} has {player_count} players; "
                f"maximum is {rules.max_players_per_team}"
            )

    lineup_picks = [pick for pick in picks if pick.squad_position <= rules.lineup_size]

    if len(lineup_picks) != rules.lineup_size:
        errors.append(f"lineup requires {rules.lineup_size} players, got {len(lineup_picks)}")

    lineup_players = [players[pick.player_id] for pick in lineup_picks if pick.player_id in players]
    lineup_position_counts: Counter[Position] = Counter(
        player.position for player in lineup_players
    )

    for position_rule in rules.position_rules:
        actual_count = lineup_position_counts[position_rule.position]

        if not (position_rule.lineup_minimum <= actual_count <= position_rule.lineup_maximum):
            errors.append(
                f"lineup requires {position_rule.lineup_minimum}-"
                f"{position_rule.lineup_maximum} "
                f"{position_rule.position.value}s, got {actual_count}"
            )

    captains = [pick for pick in picks if pick.is_captain]
    vice_captains = [pick for pick in picks if pick.is_vice_captain]

    if len(captains) != 1:
        errors.append("squad requires exactly one captain")
    elif captains[0].squad_position > rules.lineup_size:
        errors.append("captain must be in the starting lineup")

    if len(vice_captains) != 1:
        errors.append("squad requires exactly one vice-captain")
    elif vice_captains[0].squad_position > rules.lineup_size:
        errors.append("vice-captain must be in the starting lineup")

    if (
        len(captains) == 1
        and len(vice_captains) == 1
        and captains[0].player_id == vice_captains[0].player_id
    ):
        errors.append("captain and vice-captain must be different players")

    return tuple(errors)
