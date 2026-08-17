"""Render weekly planning results as a human-readable Markdown report."""

from datetime import UTC, datetime

from project_ted.fpl import (
    PlanningContext,
    Player,
    Position,
    Team,
)
from project_ted.planning import (
    AgentOutcome,
    AgentProvider,
    GameweekPlan,
    WeeklyRun,
)

_PROVIDER_LABEL = {
    AgentProvider.OPENAI: "OpenAI",
    AgentProvider.ANTHROPIC: "Anthropic",
}

_POSITION_ORDER = {
    Position.GOALKEEPER: 0,
    Position.DEFENDER: 1,
    Position.MIDFIELDER: 2,
    Position.FORWARD: 3,
}


def render_weekly_report(
    run: WeeklyRun,
    context: PlanningContext,
) -> str:
    """Render one weekly run using names and prices from its FPL context."""

    player_by_id = {player.id: player for player in context.players}
    team_by_id = {team.id: team for team in context.teams}

    _validate_report_inputs(
        run,
        context,
        player_by_id,
        team_by_id,
    )

    lines = [
        f"# Project Ted — Gameweek {run.gameweek}",
        "",
        f"**Season:** {run.season}",
        f"**Deadline:** {_format_datetime(run.deadline_at)}",
        f"**Generated:** {_format_datetime(run.created_at)}",
        f"**Run status:** {run.status.value.capitalize()}",
        f"**Run ID:** `{run.run_id}`",
    ]

    for outcome in run.outcomes:
        lines.extend(
            _render_outcome(
                outcome,
                player_by_id,
                team_by_id,
            )
        )

    lines.extend(
        _render_comparison(
            run,
            context,
            player_by_id,
        )
    )

    return "\n".join(lines).rstrip() + "\n"


def _validate_report_inputs(
    run: WeeklyRun,
    context: PlanningContext,
    player_by_id: dict[int, Player],
    team_by_id: dict[int, Team],
) -> None:
    context_matches = (
        run.season == context.season
        and run.gameweek == context.target_gameweek.id
        and run.deadline_at == context.target_gameweek.deadline_at
    )

    if not context_matches:
        raise ValueError("weekly run and planning context must match")

    referenced_player_ids: set[int] = set()

    for outcome in run.outcomes:
        if outcome.plan is not None:
            referenced_player_ids.update(outcome.plan.squad)

    unknown_player_ids = sorted(referenced_player_ids - set(player_by_id))

    if unknown_player_ids:
        formatted_ids = ", ".join(str(player_id) for player_id in unknown_player_ids)
        raise ValueError(f"report cannot resolve player IDs: {formatted_ids}")

    referenced_team_ids = {player_by_id[player_id].team_id for player_id in referenced_player_ids}
    unknown_team_ids = sorted(referenced_team_ids - set(team_by_id))

    if unknown_team_ids:
        formatted_ids = ", ".join(str(team_id) for team_id in unknown_team_ids)
        raise ValueError(f"report cannot resolve team IDs: {formatted_ids}")


def _render_outcome(
    outcome: AgentOutcome,
    player_by_id: dict[int, Player],
    team_by_id: dict[int, Team],
) -> list[str]:
    provider_name = _PROVIDER_LABEL[outcome.provider]
    lines = [
        "",
        f"## {provider_name} — {outcome.model}",
        "",
    ]

    plan = outcome.plan

    if plan is None:
        lines.extend(
            [
                "**Status:** Failed",
                "",
                f"**Error:** {outcome.error}",
            ]
        )
        return lines

    lines.extend(
        [
            "**Status:** Succeeded",
            "",
            "### Starting XI",
            "",
            "| Player | Club | Position | Price |",
            "|---|---|---|---:|",
        ]
    )

    starting_players = sorted(
        (player_by_id[player_id] for player_id in plan.starting_xi),
        key=lambda player: _POSITION_ORDER[player.position],
    )

    for player in starting_players:
        lines.append(
            _starting_player_row(
                player,
                plan,
                team_by_id,
            )
        )

    lines.extend(
        [
            "",
            "### Bench",
            "",
            "| Priority | Player | Club | Position | Price |",
            "|---:|---|---|---|---:|",
        ]
    )

    for priority, player_id in enumerate(
        plan.bench,
        start=1,
    ):
        lines.append(
            _bench_player_row(
                priority,
                player_by_id[player_id],
                team_by_id,
            )
        )

    squad_cost = sum(player_by_id[player_id].price_tenths for player_id in plan.squad)

    lines.extend(
        [
            "",
            f"**Squad cost:** {_format_price(squad_cost)}",
            "",
            "### Rationale",
            "",
            plan.rationale,
            "",
            "### Risks",
            "",
        ]
    )

    if plan.risks:
        lines.extend(f"- {risk}" for risk in plan.risks)
    else:
        lines.append("- No additional risks supplied.")

    return lines


def _starting_player_row(
    player: Player,
    plan: GameweekPlan,
    team_by_id: dict[int, Team],
) -> str:
    marker = ""

    if player.id == plan.captain_id:
        marker = " (C)"
    elif player.id == plan.vice_captain_id:
        marker = " (VC)"

    return (
        f"| {_escape_table_cell(player.name)}{marker} "
        f"| {_escape_table_cell(team_by_id[player.team_id].short_name)} "
        f"| {player.position.value} "
        f"| {_format_price(player.price_tenths)} |"
    )


def _bench_player_row(
    priority: int,
    player: Player,
    team_by_id: dict[int, Team],
) -> str:
    return (
        f"| {priority} "
        f"| {_escape_table_cell(player.name)} "
        f"| {_escape_table_cell(team_by_id[player.team_id].short_name)} "
        f"| {player.position.value} "
        f"| {_format_price(player.price_tenths)} |"
    )


def _render_comparison(
    run: WeeklyRun,
    context: PlanningContext,
    player_by_id: dict[int, Player],
) -> list[str]:
    outcome_by_provider = {outcome.provider: outcome for outcome in run.outcomes}
    openai_plan = outcome_by_provider[AgentProvider.OPENAI].plan
    anthropic_plan = outcome_by_provider[AgentProvider.ANTHROPIC].plan

    if openai_plan is None or anthropic_plan is None:
        return []

    openai_squad = set(openai_plan.squad)
    anthropic_squad = set(anthropic_plan.squad)
    shared_players = openai_squad & anthropic_squad
    openai_only = openai_squad - anthropic_squad
    anthropic_only = anthropic_squad - openai_squad

    openai_captain = player_by_id[openai_plan.captain_id].name
    anthropic_captain = player_by_id[anthropic_plan.captain_id].name

    return [
        "",
        "## Agent comparison",
        "",
        (f"**Shared squad picks:** {len(shared_players)} of {context.rules.squad_size}"),
        "",
        (f"**OpenAI only:** {_format_player_names(openai_only, player_by_id)}"),
        "",
        (f"**Anthropic only:** {_format_player_names(anthropic_only, player_by_id)}"),
        "",
        (f"**Captains:** OpenAI — {openai_captain}; Anthropic — {anthropic_captain}"),
    ]


def _format_player_names(
    player_ids: set[int],
    player_by_id: dict[int, Player],
) -> str:
    names = sorted(player_by_id[player_id].name for player_id in player_ids)

    return ", ".join(names) if names else "None"


def _format_price(price_tenths: int) -> str:
    return f"£{price_tenths / 10:.1f}m"


def _format_datetime(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _escape_table_cell(value: str) -> str:
    return " ".join(value.split()).replace("|", r"\|")
