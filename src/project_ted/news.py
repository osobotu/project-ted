import os
from datetime import UTC, datetime

from langchain_tavily import TavilySearch
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

_ALLOWED_DOMAINS = (
    "premierleague.com",
    "bbc.co.uk",
    "bbc.com",
    "skysports.com",
    "theguardian.com",
    "fantasyfootballscout.co.uk",
)

_QUERY_PREFIX = "Fantasy Premier League or English Premier League"
_MAX_RESULTS = 5


class NewsSearchError(RuntimeError):
    """Report that reliable current football news could not be obtained."""


class NewsArticle(BaseModel):
    """A normalized source returned to both planning agents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    url: HttpUrl
    summary: str = Field(min_length=1)
    relevance: float


class FootballNews(BaseModel):
    """The immutable result of one scoped football-news query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    searched_at: datetime
    articles: tuple[NewsArticle, ...]


class _RawArticle(BaseModel):
    title: str
    url: HttpUrl
    content: str
    score: float


class _RawSearchResponse(BaseModel):
    results: list[_RawArticle]


class FootballNewsSearch:
    """Provide recent trusted football news with a per-run query cache."""

    def __init__(self) -> None:
        if not os.environ.get("TAVILY_API_KEY"):
            raise NewsSearchError("TAVILY_API_KEY is not configured")

        try:
            self._tool = TavilySearch(
                max_results=_MAX_RESULTS,
                topic="news",
                search_depth="basic",
                time_range="week",
                include_domains=list(_ALLOWED_DOMAINS),
                include_answer=False,
                include_raw_content=False,
                include_images=False,
                include_image_descriptions=False,
                include_favicon=False,
                include_usage=False,
                auto_parameters=False,
                exact_match=False,
            )
        except Exception as error:
            raise NewsSearchError("Could not configure football-news search") from error

        self._cache: dict[str, FootballNews] = {}

    def search(self, query: str) -> FootballNews:
        """Return recent trusted sources relevant to one FPL question."""

        normalized_query = " ".join(query.split())

        if not normalized_query:
            raise ValueError("news query must not be empty")

        cache_key = normalized_query.casefold()
        cached_result = self._cache.get(cache_key)

        if cached_result is not None:
            return cached_result

        scoped_query = f"{_QUERY_PREFIX} {normalized_query}"

        try:
            raw_result = self._tool.invoke({"query": scoped_query})
            parsed_result = _RawSearchResponse.model_validate(raw_result)
        except Exception as error:
            raise NewsSearchError("Could not search current football news") from error

        result = FootballNews(
            query=normalized_query,
            searched_at=datetime.now(UTC),
            articles=tuple(
                NewsArticle(
                    title=article.title,
                    url=article.url,
                    summary=article.content,
                    relevance=article.score,
                )
                for article in parsed_result.results
            ),
        )
        self._cache[cache_key] = result

        return result
