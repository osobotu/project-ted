"""Construct provider models for the shared planning harness."""

import os
from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from project_ted.planning import AgentProvider

_DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
_OPENAI_MODEL_ENV = "PROJECT_TED_OPENAI_MODEL"

_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
_ANTHROPIC_MODEL_ENV = "PROJECT_TED_ANTHROPIC_MODEL"


class ProviderConfigurationError(RuntimeError):
    """Report missing or invalid model-provider configuration."""


@dataclass(frozen=True, slots=True)
class ProviderModel:
    """Bind a chat model to the identity recorded in planning results."""

    provider: AgentProvider
    model_name: str
    chat_model: BaseChatModel


def create_openai_model() -> ProviderModel:
    """Return the configured OpenAI model and its stable identity."""

    model_name = _configured_model(
        api_key_env="OPENAI_API_KEY",
        model_env=_OPENAI_MODEL_ENV,
        default_model=_DEFAULT_OPENAI_MODEL,
    )

    chat_model = ChatOpenAI(
        model=model_name,
        reasoning_effort="medium",
        max_retries=2,
        timeout=120.0,
        use_responses_api=True,
    )

    return ProviderModel(
        provider=AgentProvider.OPENAI,
        model_name=model_name,
        chat_model=chat_model,
    )


def create_anthropic_model() -> ProviderModel:
    """Return the configured Anthropic model and its stable identity."""

    model_name = _configured_model(
        api_key_env="ANTHROPIC_API_KEY",
        model_env=_ANTHROPIC_MODEL_ENV,
        default_model=_DEFAULT_ANTHROPIC_MODEL,
    )

    chat_model = ChatAnthropic(
        model_name=model_name,
        effort="medium",
        max_retries=2,
        timeout=120.0,
        stop=None,
    )

    return ProviderModel(
        provider=AgentProvider.ANTHROPIC,
        model_name=model_name,
        chat_model=chat_model,
    )


def _configured_model(
    *,
    api_key_env: str,
    model_env: str,
    default_model: str,
) -> str:
    """Validate provider configuration and return its selected model."""

    if not os.environ.get(api_key_env, "").strip():
        raise ProviderConfigurationError(f"{api_key_env} is not configured")

    model_name = os.environ.get(
        model_env,
        default_model,
    ).strip()

    if not model_name:
        raise ProviderConfigurationError(f"{model_env} must not be empty")

    return model_name
