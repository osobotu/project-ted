import pytest
from langchain_tavily import TavilySearch

from project_ted.news import (
    FootballNewsSearch,
    NewsSearchError,
)


def configure_fake_tavily(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[TavilySearch], list[str]]:
    observed_tools: list[TavilySearch] = []
    observed_queries: list[str] = []

    def fake_invoke(
        tool: TavilySearch,
        tool_input: dict[str, str],
    ) -> dict[str, object]:
        observed_tools.append(tool)
        observed_queries.append(tool_input["query"])

        return {
            "query": tool_input["query"],
            "results": [
                {
                    "title": "Arsenal provide injury update",
                    "url": ("https://www.premierleague.com/news/arsenal-injury-update"),
                    "content": (
                        "Arsenal supplied an update before the next Premier League fixture."
                    ),
                    "score": 0.91,
                }
            ],
        }

    monkeypatch.setattr(
        TavilySearch,
        "invoke",
        fake_invoke,
    )

    return observed_tools, observed_queries


def test_normalizes_and_restricts_football_news(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    observed_tools, observed_queries = configure_fake_tavily(monkeypatch)

    search = FootballNewsSearch()
    result = search.search("  Arsenal   injury updates  ")

    assert result.query == "Arsenal injury updates"
    assert len(result.articles) == 1

    article = result.articles[0]
    assert article.title == "Arsenal provide injury update"
    assert str(article.url) == ("https://www.premierleague.com/news/arsenal-injury-update")
    assert article.relevance == 0.91

    assert observed_queries == [
        ("Fantasy Premier League or English Premier League Arsenal injury updates")
    ]

    tool = observed_tools[0]
    assert tool.topic == "news"
    assert tool.search_depth == "basic"
    assert tool.time_range == "week"
    assert tool.max_results == 5
    assert tool.auto_parameters is False
    assert tool.include_domains == [
        "premierleague.com",
        "bbc.co.uk",
        "bbc.com",
        "skysports.com",
        "theguardian.com",
        "fantasyfootballscout.co.uk",
    ]


def test_reuses_an_identical_query_during_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    _, observed_queries = configure_fake_tavily(monkeypatch)

    search = FootballNewsSearch()

    first = search.search("Liverpool injuries")
    second = search.search("  liverpool   INJURIES  ")

    assert first is second
    assert len(observed_queries) == 1


def test_requires_a_tavily_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(
        NewsSearchError,
        match="TAVILY_API_KEY is not configured",
    ):
        FootballNewsSearch()


def test_rejects_an_empty_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    configure_fake_tavily(monkeypatch)

    search = FootballNewsSearch()

    with pytest.raises(
        ValueError,
        match="news query must not be empty",
    ):
        search.search("   ")


def test_hides_invalid_tavily_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")

    def malformed_response(
        tool: TavilySearch,
        tool_input: dict[str, str],
    ) -> str:
        return "Tavily failed"

    monkeypatch.setattr(
        TavilySearch,
        "invoke",
        malformed_response,
    )

    search = FootballNewsSearch()

    with pytest.raises(
        NewsSearchError,
        match="Could not search current football news",
    ):
        search.search("Chelsea team news")
