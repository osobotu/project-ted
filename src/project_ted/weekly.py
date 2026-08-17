"""Run both planning providers and preserve their independent outcomes."""

from datetime import UTC, datetime
from uuid import uuid4

import langsmith as ls

from project_ted.agent import plan_gameweek
from project_ted.fpl import PlanningContext
from project_ted.news import FootballNewsSearch
from project_ted.planning import AgentOutcome, WeeklyRun
from project_ted.providers import (
    ProviderModel,
    create_anthropic_model,
    create_openai_model,
)


def run_weekly_planning(context: PlanningContext, news: FootballNewsSearch) -> WeeklyRun:
    """Run both providers against one shared context and return their outcomes."""

    run_id = uuid4()
    created_at = datetime.now(UTC)

    openai_model = create_openai_model()
    anthropic_model = create_anthropic_model()

    with ls.trace(
        "weekly-fpl-planning",
        run_type="chain",
        run_id=run_id,
        inputs={
            "season": context.season,
            "gameweek": context.target_gameweek.id,
        },
        tags=["project-ted", "weekly-planning"],
        metadata={
            "run_id": str(run_id),
            "deadline": context.target_gameweek.deadline_at.isoformat(),
        },
    ) as weekly_trace:
        outcomes = (
            _run_provider(openai_model, context, news),
            _run_provider(anthropic_model, context, news),
        )

        result = WeeklyRun(
            run_id=run_id,
            season=context.season,
            gameweek=context.target_gameweek.id,
            created_at=created_at,
            deadline_at=context.target_gameweek.deadline_at,
            outcomes=outcomes,
        )

        weekly_trace.end(
            outputs=result.model_dump(mode="json"),
        )

    return result


def _run_provider(
    configured_model: ProviderModel,
    context: PlanningContext,
    news: FootballNewsSearch,
) -> AgentOutcome:
    with ls.trace(
        f"{configured_model.provider.value}-planning",
        run_type="chain",
        inputs={
            "provider": configured_model.provider.value,
            "model": configured_model.model_name,
        },
        tags=[
            "project-ted",
            configured_model.provider.value,
        ],
        metadata={
            "provider": configured_model.provider.value,
            "model": configured_model.model_name,
        },
    ) as provider_trace:
        try:
            plan = plan_gameweek(
                configured_model.chat_model,
                context,
                news,
            )
            outcome = AgentOutcome(
                provider=configured_model.provider,
                model=configured_model.model_name,
                plan=plan,
            )
        except Exception as error:
            outcome = AgentOutcome(
                provider=configured_model.provider,
                model=configured_model.model_name,
                error=f"{type(error).__name__}: {error}",
            )

        provider_trace.end(
            outputs=outcome.model_dump(mode="json"),
        )

    return outcome
