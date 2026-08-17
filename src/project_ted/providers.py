"""OpenAI implementation of LLM provider"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

_DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
_OPENAI_MODEL_ENV = "PROJECT_TED_OPENAI_MODEL"

_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
_ANTHROPIC_MODEL_ENV = "PROJECT_TED_ANTHROPIC_MODEL"


class ProviderConfigurationError(RuntimeError):
    """Report missing or invalid model-provider configuration."""


def create_openai_model() -> BaseChatModel:
    """Return the configured OpenAI model for the shared planning harness."""

    model_name = _configured_model(
        api_key_env="OPENAI_API_KEY",
        model_env=_OPENAI_MODEL_ENV,
        default_model=_DEFAULT_OPENAI_MODEL,
    )

    return ChatOpenAI(
        model=model_name,
        reasoning_effort="medium",
        max_retries=2,
        timeout=120.0,
        use_responses_api=True,
    )


def create_anthropic_model() -> BaseChatModel:
    """Return the configured Anthropic model for the shared planning harness."""

    model_name = _configured_model(
        api_key_env="ANTHROPIC_API_KEY",
        model_env=_ANTHROPIC_MODEL_ENV,
        default_model=_DEFAULT_ANTHROPIC_MODEL,
    )

    return ChatAnthropic(
        model_name=model_name,
        effort="medium",
        max_retries=2,
        timeout=120.0,
        stop=None,
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
