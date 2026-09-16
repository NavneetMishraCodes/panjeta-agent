"""Deterministic offline tests for LLM provider selection (create_llm)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from src.llm import (
    PROVIDER_GOOGLE,
    PROVIDER_OPENROUTER,
    UnknownLLMProviderError,
    create_llm,
)
from src.llm.google import GoogleLLM
from src.llm.openrouter import OpenRouterLLM


class FactoryTest(unittest.TestCase):
    def _env(self, **extra):
        # Merge on top of the existing environment: clearing it entirely
        # breaks SDK client construction on some platforms (SSL context
        # init depends on OS env vars), and tests should not do that.
        env = {
            "OPENROUTER_API_KEY": "k-openrouter",
            "OPENROUTER_MODEL": "m-openrouter",
            "GOOGLE_API_KEY": "k-google",
            "GOOGLE_MODEL": "m-google",
        }
        env.update(extra)
        return mock.patch.dict(os.environ, env)

    def test_creates_openrouter_provider(self):
        with self._env():
            llm = create_llm("openrouter")
        self.assertIsInstance(llm, OpenRouterLLM)

    def test_creates_google_provider(self):
        with self._env():
            llm = create_llm("google")
        self.assertIsInstance(llm, GoogleLLM)

    def test_open_router_alias(self):
        with self._env():
            llm = create_llm("open_router")
        self.assertIsInstance(llm, OpenRouterLLM)

    def test_gemini_alias(self):
        with self._env():
            llm = create_llm("gemini")
        self.assertIsInstance(llm, GoogleLLM)

    def test_case_and_whitespace_insensitive(self):
        with self._env():
            llm = create_llm("  GOOGLE  ")
        self.assertIsInstance(llm, GoogleLLM)

    def test_unknown_provider_raises(self):
        with self._env():
            with self.assertRaises(UnknownLLMProviderError):
                create_llm("anthropic")

    def test_provider_constant_names(self):
        self.assertEqual(PROVIDER_OPENROUTER, "openrouter")
        self.assertEqual(PROVIDER_GOOGLE, "google")

    def test_forwarded_kwargs_reach_the_provider(self):
        with self._env():
            llm = create_llm("google", api_key="explicit", model="custom-model")
        self.assertEqual(llm._model, "custom-model")

    def test_missing_provider_config_raises_generic_config_error(self):
        # Constructing a real GoogleLLM without any key/model must surface
        # as a configuration problem the entry point can report generically.
        # The empty strings mimic "unset" without wiping unrelated env vars.
        with mock.patch.dict(
            os.environ, {"GOOGLE_API_KEY": "", "GOOGLE_MODEL": ""}
        ):
            with self.assertRaises(Exception) as ctx:
                create_llm("google")
        from src.llm import LLMConfigError

        self.assertIsInstance(ctx.exception, LLMConfigError)


if __name__ == "__main__":
    unittest.main()