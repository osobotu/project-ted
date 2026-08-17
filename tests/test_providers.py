from typing import cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

import project_ted.providers as provider_module
from project_ted.providers import (
    ProviderConfigurationError,
    create_anthropic_model,
    create_openai_model,
)


def test_requires_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(
        ProviderConfigurationError,
        match="OPENAI_API_KEY is not configured",
    ):
        create_openai_model()


@pytest.mark.parametrize(
    ("configured_model", "expected_model"),
    [
        (None, "gpt-5.6-terra"),
        ("gpt-test", "gpt-test"),
    ],
)
def test_configures_openai_model(
    monkeypatch: pytest.MonkeyPatch,
    configured_model: str | None,
    expected_model: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    if configured_model is None:
        monkeypatch.delenv("PROJECT_TED_OPENAI_MODEL", raising=False)
    else:
        monkeypatch.setenv(
            "PROJECT_TED_OPENAI_MODEL",
            configured_model,
        )

    captured: dict[str, object] = {}
    expected_instance = cast(BaseChatModel, object())

    def fake_chat_openai(**kwargs: object) -> BaseChatModel:
        captured.update(kwargs)
        return expected_instance

    monkeypatch.setattr(
        provider_module,
        "ChatOpenAI",
        fake_chat_openai,
    )

    result = create_openai_model()

    assert result is expected_instance
    assert captured == {
        "model": expected_model,
        "reasoning_effort": "medium",
        "max_retries": 2,
        "timeout": 120.0,
        "use_responses_api": True,
    }


def test_rejects_blank_openai_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PROJECT_TED_OPENAI_MODEL", "   ")

    with pytest.raises(
        ProviderConfigurationError,
        match="PROJECT_TED_OPENAI_MODEL must not be empty",
    ):
        create_openai_model()


def test_requires_anthropic_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(
        ProviderConfigurationError,
        match="ANTHROPIC_API_KEY is not configured",
    ):
        create_anthropic_model()


@pytest.mark.parametrize(
    ("configured_model", "expected_model"),
    [
        (None, "claude-sonnet-5"),
        ("claude-test", "claude-test"),
    ],
)
def test_configures_anthropic_model(
    monkeypatch: pytest.MonkeyPatch,
    configured_model: str | None,
    expected_model: str,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    if configured_model is None:
        monkeypatch.delenv(
            "PROJECT_TED_ANTHROPIC_MODEL",
            raising=False,
        )
    else:
        monkeypatch.setenv(
            "PROJECT_TED_ANTHROPIC_MODEL",
            configured_model,
        )

    captured: dict[str, object] = {}
    expected_instance = cast(BaseChatModel, object())

    def fake_chat_anthropic(**kwargs: object) -> BaseChatModel:
        captured.update(kwargs)
        return expected_instance

    monkeypatch.setattr(
        provider_module,
        "ChatAnthropic",
        fake_chat_anthropic,
    )

    result = create_anthropic_model()

    assert result is expected_instance
    assert captured == {
        "model_name": expected_model,
        "effort": "medium",
        "max_retries": 2,
        "timeout": 120.0,
        "stop": None,
    }


def test_rejects_blank_anthropic_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("PROJECT_TED_ANTHROPIC_MODEL", "   ")

    with pytest.raises(
        ProviderConfigurationError,
        match="PROJECT_TED_ANTHROPIC_MODEL must not be empty",
    ):
        create_anthropic_model()
