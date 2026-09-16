"""Provider factory: build any registered LLM from a provider name.

Keeps the Agent and the interactive entry point provider-agnostic:

    create_llm("openrouter") -> OpenRouterLLM
    create_llm("google")     -> GoogleLLM

New providers register here (with aliases), not inside Agent or main.
"""

from __future__ import annotations

from src.llm.base import BaseLLM
from src.llm.google import GoogleLLM
from src.llm.openrouter import OpenRouterLLM

PROVIDER_OPENROUTER = "openrouter"
PROVIDER_GOOGLE = "google"

# Canonical provider name -> accepted names (lowercased).
_ALIASES = {
    "openrouter": PROVIDER_OPENROUTER,
    "open_router": PROVIDER_OPENROUTER,
    "google": PROVIDER_GOOGLE,
    "gemini": PROVIDER_GOOGLE,
}

SUPPORTED_PROVIDERS = (PROVIDER_OPENROUTER, PROVIDER_GOOGLE)


class LLMProviderError(RuntimeError):
    """Base class for provider-selection failures."""


class UnknownLLMProviderError(LLMProviderError):
    """Raised when create_llm is given an unrecognized provider name."""


def create_llm(provider: str, **kwargs) -> BaseLLM:
    """Build an LLM client for the given provider name.

    Args:
        provider: Provider name; case- and whitespace-insensitive
            ("openrouter", "google", or aliases "open_router"/"gemini").
        **kwargs: Forwarded to the provider constructor (e.g. api_key,
            model).

    Returns:
        A configured BaseLLM instance.

    Raises:
        UnknownLLMProviderError: If the provider name is not supported.
        LLMConfigError: If the selected provider's configuration is
            missing or incomplete.
    """
    normalized = (provider or "").strip().lower()
    canonical = _ALIASES.get(normalized)

    if canonical == PROVIDER_OPENROUTER:
        return OpenRouterLLM(**kwargs)
    if canonical == PROVIDER_GOOGLE:
        return GoogleLLM(**kwargs)
    raise UnknownLLMProviderError(
        f"Unknown LLM provider {provider!r}. Supported providers: "
        f"{', '.join(SUPPORTED_PROVIDERS)}."
    )