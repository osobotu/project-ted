"""OpenAI implementation of LLM provider"""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

_DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
_OPENAI_MODEL_ENV = "PROJECT_TED_OPENAI_MODEL"


class ProviderConfigurationError(RuntimeError):
    """Report missing or invalid model-provider configuration."""


def create_openai_model() -> BaseChatModel:
    """Return the configured OpenAI model for the shared planning harness."""

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise ProviderConfigurationError("OPENAI_API_KEY is not configured")

    model_name = os.environ.get(
        _OPENAI_MODEL_ENV,
        _DEFAULT_OPENAI_MODEL,
    ).strip()

    if not model_name:
        raise ProviderConfigurationError("PROJECT_TED_OPENAI_MODEL must not be empty")

    return ChatOpenAI(
        model=model_name,
        reasoning_effort="medium",
        max_retries=2,
        timeout=120.0,
        use_responses_api=True,
    )
