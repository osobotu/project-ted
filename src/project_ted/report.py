"""Render one weekly planning result for email and archival use."""

from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape

from project_ted.fpl import PlanningContext, Player, Position, Team
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

_BODY_STYLE = "margin:0;background:#f3f4f6;font-family:Arial,Helvetica,sans-serif;color:#172033;"
_CONTAINER_STYLE = (
    "max-width:680px;margin:0 auto;background:#ffffff;border-radius:16px;overflow:hidden;"
)
_HEADER_STYLE = "background:#101828;color:#ffffff;padding:32px 28px;"
_CONTENT_STYLE = "padding:24px 28px;"
_CARD_STYLE = "border:1px solid #e4e7ec;border-radius:12px;padding:20px;margin:0 0 24px 0;"
_TABLE_STYLE = "width:100%;border-collapse:collapse;margin:12px 0 20px 0;"
_TH_STYLE = (
    "background:#f2f4f7;color:#475467;font-size:12px;"
    "text-align:left;padding:10px;border-bottom:1px solid #d0d5dd;"
)
_TD_STYLE = "font-size:14px;padding:10px;border-bottom:1px solid #eaecf0;"
_SECTION_STYLE = "font-size:16px;color:#101828;margin:20px 0 8px 0;"
_MUTED_STYLE = "color:#667085;font-size:13px;"
_ERROR_STYLE = "background:#fef3f2;color:#b42318;border-radius:8px;padding:14px;font-size:14px;"


@dataclass(frozen=True, slots=True)
class RenderedReport:
    """All representations of one validated weekly report."""

    markdown: str
    text: str
    html: str


def render_weekly_report(
    run: WeeklyRun,
    context: PlanningContext,
) -> RenderedReport:
    """Render one validated weekly run as Markdown, plain text, and HTML."""

    player_by_id = {player.id: player for player in context.players}
    team_by_id = {team.id: team for team in context.teams}

    _validate_report_inputs(
        run,
        context,
        player_by_id,
        team_by_id,
    )

    return RenderedReport(
        markdown=_render_markdown(
            run,
            context,
            player_by_id,
            team_by_id,
        ),
        text=_render_text(
            run,
            context,
            player_by_id,
            team_by_id,
        ),
        html=_render_html(
            run,
            context,
            player_by_id,
            team_by_id,
        ),
    )


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


def _render_markdown(
    run: WeeklyRun,
    context: PlanningContext,
    player_by_id: dict[int, Player],
    team_by_id: dict[int, Team],
) -> str:
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
            _render_markdown_outcome(
                outcome,
                player_by_id,
                team_by_id,
            )
        )

    lines.extend(
        _render_markdown_comparison(
            run,
            context,
            player_by_id,
        )
    )

    return "\n".join(lines).rstrip() + "\n"


def _render_markdown_outcome(
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

    for player in _starting_players(plan, player_by_id):
        lines.append(
            "| "
            f"{_escape_markdown_cell(player.name)}"
            f"{_captain_suffix(player.id, plan)} | "
            f"{_escape_markdown_cell(team_by_id[player.team_id].short_name)} | "
            f"{player.position.value} | "
            f"{_format_price(player.price_tenths)} |"
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

    for priority, player_id in enumerate(plan.bench, start=1):
        player = player_by_id[player_id]
        team = team_by_id[player.team_id]

        lines.append(
            f"| {priority} | "
            f"{_escape_markdown_cell(player.name)} | "
            f"{_escape_markdown_cell(team.short_name)} | "
            f"{player.position.value} | "
            f"{_format_price(player.price_tenths)} |"
        )

    lines.extend(
        [
            "",
            f"**Squad cost:** {_squad_cost(plan, player_by_id)}",
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


def _render_markdown_comparison(
    run: WeeklyRun,
    context: PlanningContext,
    player_by_id: dict[int, Player],
) -> list[str]:
    comparison = _comparison(run, player_by_id)

    if comparison is None:
        return []

    shared_count, openai_only, anthropic_only, captains = comparison

    return [
        "",
        "## Agent comparison",
        "",
        f"**Shared squad picks:** {shared_count} of {context.rules.squad_size}",
        "",
        f"**OpenAI only:** {openai_only}",
        "",
        f"**Anthropic only:** {anthropic_only}",
        "",
        f"**Captains:** {captains}",
    ]


def _render_text(
    run: WeeklyRun,
    context: PlanningContext,
    player_by_id: dict[int, Player],
    team_by_id: dict[int, Team],
) -> str:
    lines = [
        f"PROJECT TED — GAMEWEEK {run.gameweek}",
        "",
        f"Season: {run.season}",
        f"Deadline: {_format_datetime(run.deadline_at)}",
        f"Generated: {_format_datetime(run.created_at)}",
        f"Run status: {run.status.value.capitalize()}",
        f"Run ID: {run.run_id}",
    ]

    for outcome in run.outcomes:
        provider_name = _PROVIDER_LABEL[outcome.provider]
        lines.extend(
            [
                "",
                "=" * 60,
                f"{provider_name.upper()} — {outcome.model}",
                "=" * 60,
            ]
        )

        plan = outcome.plan

        if plan is None:
            lines.extend(
                [
                    "Status: Failed",
                    f"Error: {outcome.error}",
                ]
            )
            continue

        lines.extend(
            [
                "Status: Succeeded",
                "",
                "STARTING XI",
            ]
        )

        for player in _starting_players(plan, player_by_id):
            team = team_by_id[player.team_id]
            lines.append(
                f"- {player.position.value}: "
                f"{player.name}{_captain_suffix(player.id, plan)} — "
                f"{team.short_name} — {_format_price(player.price_tenths)}"
            )

        lines.extend(
            [
                "",
                "BENCH",
            ]
        )

        for priority, player_id in enumerate(plan.bench, start=1):
            player = player_by_id[player_id]
            team = team_by_id[player.team_id]
            lines.append(
                f"{priority}. {player.name} — {team.short_name} — "
                f"{player.position.value} — {_format_price(player.price_tenths)}"
            )

        lines.extend(
            [
                "",
                f"Squad cost: {_squad_cost(plan, player_by_id)}",
                "",
                "RATIONALE",
                plan.rationale,
                "",
                "RISKS",
            ]
        )

        if plan.risks:
            lines.extend(f"- {risk}" for risk in plan.risks)
        else:
            lines.append("- No additional risks supplied.")

    comparison = _comparison(run, player_by_id)

    if comparison is not None:
        shared_count, openai_only, anthropic_only, captains = comparison
        lines.extend(
            [
                "",
                "=" * 60,
                "AGENT COMPARISON",
                "=" * 60,
                (f"Shared squad picks: {shared_count} of {context.rules.squad_size}"),
                f"OpenAI only: {openai_only}",
                f"Anthropic only: {anthropic_only}",
                f"Captains: {captains}",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def _render_html(
    run: WeeklyRun,
    context: PlanningContext,
    player_by_id: dict[int, Player],
    team_by_id: dict[int, Team],
) -> str:
    status_color = {
        "succeeded": "#067647",
        "partial": "#b54708",
        "failed": "#b42318",
    }[run.status.value]

    outcome_cards = "".join(
        _render_html_outcome(
            outcome,
            player_by_id,
            team_by_id,
        )
        for outcome in run.outcomes
    )

    comparison = _render_html_comparison(
        run,
        context,
        player_by_id,
    )

    return "".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width">',
            f"<title>Project Ted — Gameweek {run.gameweek}</title>",
            "</head>",
            f'<body style="{_BODY_STYLE}">',
            '<div style="padding:24px 12px;">',
            f'<div style="{_CONTAINER_STYLE}">',
            f'<div style="{_HEADER_STYLE}">',
            (
                '<div style="font-size:13px;color:#98a2b3;'
                'letter-spacing:1px;text-transform:uppercase;">'
                "Weekly FPL plan"
                "</div>"
            ),
            (
                '<h1 style="margin:8px 0 16px 0;font-size:28px;'
                'line-height:1.2;">'
                f"Project Ted — Gameweek {run.gameweek}"
                "</h1>"
            ),
            (
                f'<span style="display:inline-block;background:{status_color};'
                "color:#ffffff;border-radius:999px;padding:6px 12px;"
                'font-size:12px;font-weight:bold;">'
                f"{escape(run.status.value.capitalize())}"
                "</span>"
            ),
            "</div>",
            f'<div style="{_CONTENT_STYLE}">',
            '<table role="presentation" style="width:100%;margin-bottom:24px;">',
            _html_metadata_row("Season", run.season),
            _html_metadata_row(
                "Deadline",
                _format_datetime(run.deadline_at),
            ),
            _html_metadata_row(
                "Generated",
                _format_datetime(run.created_at),
            ),
            _html_metadata_row("Run ID", str(run.run_id)),
            "</table>",
            outcome_cards,
            comparison,
            (
                '<p style="margin:24px 0 0 0;color:#98a2b3;'
                'font-size:12px;text-align:center;">'
                "Generated by Project Ted using current FPL data."
                "</p>"
            ),
            "</div>",
            "</div>",
            "</div>",
            "</body>",
            "</html>",
        ]
    )


def _render_html_outcome(
    outcome: AgentOutcome,
    player_by_id: dict[int, Player],
    team_by_id: dict[int, Team],
) -> str:
    provider_name = _PROVIDER_LABEL[outcome.provider]
    plan = outcome.plan

    heading = "".join(
        [
            f'<div style="{_CARD_STYLE}">',
            (
                '<div style="font-size:12px;color:#667085;'
                'text-transform:uppercase;letter-spacing:0.8px;">'
                f"{escape(provider_name)} agent"
                "</div>"
            ),
            (
                '<h2 style="font-size:22px;margin:6px 0 16px 0;'
                'color:#101828;">'
                f"{escape(provider_name)}"
                '<span style="font-size:13px;color:#667085;font-weight:normal;">'
                f" — {escape(outcome.model)}"
                "</span>"
                "</h2>"
            ),
        ]
    )

    if plan is None:
        return "".join(
            [
                heading,
                (
                    '<div style="color:#b42318;font-size:13px;'
                    'font-weight:bold;margin-bottom:8px;">Failed</div>'
                ),
                f'<div style="{_ERROR_STYLE}">',
                escape(outcome.error or "Unknown provider failure"),
                "</div>",
                "</div>",
            ]
        )

    starting_rows = "".join(
        _html_player_row(
            player,
            team_by_id[player.team_id],
            _captain_suffix(player.id, plan),
        )
        for player in _starting_players(plan, player_by_id)
    )

    bench_rows = "".join(
        _html_bench_row(
            priority,
            player_by_id[player_id],
            team_by_id[player_by_id[player_id].team_id],
        )
        for priority, player_id in enumerate(plan.bench, start=1)
    )

    risks = "".join((f'<li style="margin-bottom:6px;">{escape(risk)}</li>') for risk in plan.risks)

    if not risks:
        risks = "<li>No additional risks supplied.</li>"

    return "".join(
        [
            heading,
            (
                '<div style="color:#067647;font-size:13px;'
                'font-weight:bold;margin-bottom:8px;">Succeeded</div>'
            ),
            f'<h3 style="{_SECTION_STYLE}">Starting XI</h3>',
            f'<table role="presentation" style="{_TABLE_STYLE}">',
            "<thead><tr>",
            f'<th style="{_TH_STYLE}">Player</th>',
            f'<th style="{_TH_STYLE}">Club</th>',
            f'<th style="{_TH_STYLE}">Position</th>',
            f'<th style="{_TH_STYLE}text-align:right;">Price</th>',
            "</tr></thead>",
            f"<tbody>{starting_rows}</tbody>",
            "</table>",
            f'<h3 style="{_SECTION_STYLE}">Bench</h3>',
            f'<table role="presentation" style="{_TABLE_STYLE}">',
            "<thead><tr>",
            f'<th style="{_TH_STYLE}">#</th>',
            f'<th style="{_TH_STYLE}">Player</th>',
            f'<th style="{_TH_STYLE}">Club</th>',
            f'<th style="{_TH_STYLE}">Position</th>',
            f'<th style="{_TH_STYLE}text-align:right;">Price</th>',
            "</tr></thead>",
            f"<tbody>{bench_rows}</tbody>",
            "</table>",
            (
                '<div style="background:#ecfdf3;color:#067647;'
                'border-radius:8px;padding:12px;font-weight:bold;">'
                f"Squad cost: {_squad_cost(plan, player_by_id)}"
                "</div>"
            ),
            f'<h3 style="{_SECTION_STYLE}">Rationale</h3>',
            (
                '<p style="font-size:14px;line-height:1.6;margin:0;">'
                f"{_html_text(plan.rationale)}"
                "</p>"
            ),
            f'<h3 style="{_SECTION_STYLE}">Risks</h3>',
            (
                '<ul style="font-size:14px;line-height:1.5;'
                'padding-left:20px;margin-bottom:0;">'
                f"{risks}"
                "</ul>"
            ),
            "</div>",
        ]
    )


def _render_html_comparison(
    run: WeeklyRun,
    context: PlanningContext,
    player_by_id: dict[int, Player],
) -> str:
    comparison = _comparison(run, player_by_id)

    if comparison is None:
        return ""

    shared_count, openai_only, anthropic_only, captains = comparison

    return "".join(
        [
            f'<div style="{_CARD_STYLE}background:#f9fafb;">',
            ('<h2 style="font-size:20px;margin:0 0 14px 0;color:#101828;">Agent comparison</h2>'),
            (
                f'<p style="{_MUTED_STYLE}">'
                f"<strong>Shared picks:</strong> "
                f"{shared_count} of {context.rules.squad_size}"
                "</p>"
            ),
            (f'<p style="{_MUTED_STYLE}"><strong>OpenAI only:</strong> {escape(openai_only)}</p>'),
            (
                f'<p style="{_MUTED_STYLE}">'
                f"<strong>Anthropic only:</strong> {escape(anthropic_only)}"
                "</p>"
            ),
            (
                f'<p style="{_MUTED_STYLE}margin-bottom:0;">'
                f"<strong>Captains:</strong> {escape(captains)}"
                "</p>"
            ),
            "</div>",
        ]
    )


def _html_metadata_row(label: str, value: str) -> str:
    return "".join(
        [
            "<tr>",
            (
                '<td style="padding:5px 12px 5px 0;color:#667085;'
                'font-size:13px;width:90px;">'
                f"{escape(label)}"
                "</td>"
            ),
            (f'<td style="padding:5px 0;color:#101828;font-size:13px;">{escape(value)}</td>'),
            "</tr>",
        ]
    )


def _html_player_row(
    player: Player,
    team: Team,
    captain_suffix: str,
) -> str:
    return "".join(
        [
            "<tr>",
            f'<td style="{_TD_STYLE}">',
            f"<strong>{escape(player.name)}</strong>",
            (f'<span style="color:#067647;font-weight:bold;">{escape(captain_suffix)}</span>'),
            "</td>",
            f'<td style="{_TD_STYLE}">{escape(team.short_name)}</td>',
            f'<td style="{_TD_STYLE}">{escape(player.position.value)}</td>',
            (f'<td style="{_TD_STYLE}text-align:right;">{_format_price(player.price_tenths)}</td>'),
            "</tr>",
        ]
    )


def _html_bench_row(
    priority: int,
    player: Player,
    team: Team,
) -> str:
    return "".join(
        [
            "<tr>",
            f'<td style="{_TD_STYLE}">{priority}</td>',
            f'<td style="{_TD_STYLE}"><strong>{escape(player.name)}</strong></td>',
            f'<td style="{_TD_STYLE}">{escape(team.short_name)}</td>',
            f'<td style="{_TD_STYLE}">{escape(player.position.value)}</td>',
            (f'<td style="{_TD_STYLE}text-align:right;">{_format_price(player.price_tenths)}</td>'),
            "</tr>",
        ]
    )


def _comparison(
    run: WeeklyRun,
    player_by_id: dict[int, Player],
) -> tuple[int, str, str, str] | None:
    outcome_by_provider = {outcome.provider: outcome for outcome in run.outcomes}
    openai_plan = outcome_by_provider[AgentProvider.OPENAI].plan
    anthropic_plan = outcome_by_provider[AgentProvider.ANTHROPIC].plan

    if openai_plan is None or anthropic_plan is None:
        return None

    openai_squad = set(openai_plan.squad)
    anthropic_squad = set(anthropic_plan.squad)

    shared_players = openai_squad & anthropic_squad
    openai_only = openai_squad - anthropic_squad
    anthropic_only = anthropic_squad - openai_squad

    openai_captain = player_by_id[openai_plan.captain_id].name
    anthropic_captain = player_by_id[anthropic_plan.captain_id].name

    captains = f"OpenAI — {openai_captain}; Anthropic — {anthropic_captain}"

    return (
        len(shared_players),
        _format_player_names(openai_only, player_by_id),
        _format_player_names(anthropic_only, player_by_id),
        captains,
    )


def _starting_players(
    plan: GameweekPlan,
    player_by_id: dict[int, Player],
) -> list[Player]:
    return sorted(
        (player_by_id[player_id] for player_id in plan.starting_xi),
        key=lambda player: _POSITION_ORDER[player.position],
    )


def _captain_suffix(
    player_id: int,
    plan: GameweekPlan,
) -> str:
    if player_id == plan.captain_id:
        return " (C)"

    if player_id == plan.vice_captain_id:
        return " (VC)"

    return ""


def _squad_cost(
    plan: GameweekPlan,
    player_by_id: dict[int, Player],
) -> str:
    total = sum(player_by_id[player_id].price_tenths for player_id in plan.squad)
    return _format_price(total)


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


def _escape_markdown_cell(value: str) -> str:
    return " ".join(value.split()).replace("|", r"\|")


def _html_text(value: str) -> str:
    return escape(value).replace("\n", "<br>")
