# Project Ted

A multi-agent Fantasy Premier League planner that uses live FPL data and football news to
generate OpenAI and Anthropic team recommendations and email the resulting report.

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and `make`.

```bash
uv sync
cp .env.example .env
```

Add the required API keys and email settings to `.env`. Set `LANGSMITH_TRACING=false` if
LangSmith is not configured.

## Run locally

```bash
make dev-check
uv run --env-file .env project-ted
```

The second command uses live APIs and sends the generated report by email.
