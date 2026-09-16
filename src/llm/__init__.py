"""Panjeta LLM layer: provider-independent types + concrete providers.

Import everything through this package so callers never need to reach
into deep modules. The Agent depends only on the generic types from
``src.llm.base``; the interactive entry point selects a provider through
``create_llm``.
"""

from src.llm.base import (
    LLMConfigError,
    BaseLLM,
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
)
from src.llm.factory import (
    PROVIDER_GOOGLE,
    PROVIDER_OPENROUTER,
    LLMProviderError,
    UnknownLLMProviderError,
    create_llm,
)
from src.llm.google import GoogleConfigError, GoogleLLM, GoogleToolCallError
from src.llm.openrouter import (
    OpenRouterConfigError,
    OpenRouterLLM,
    OpenRouterToolCallError,
)

__all__ = [
    # provider-independent types and interface
    "LLMConfigError",
    "BaseLLM",
    "LLMResponse",
    "Message",
    "Role",
    "ToolCall",
    "ToolDefinition",
    # provider factory / selection
    "PROVIDER_GOOGLE",
    "PROVIDER_OPENROUTER",
    "LLMProviderError",
    "UnknownLLMProviderError",
    "create_llm",
    # OpenRouter provider
    "OpenRouterConfigError",
    "OpenRouterLLM",
    "OpenRouterToolCallError",
    # Google Gemini provider
    "GoogleConfigError",
    "GoogleLLM",
    "GoogleToolCallError",
]
