"""Offline tests for the session CLI controls (--fresh, session status/reset).

These run the real ``main()`` command paths with ``PANJETA_SESSION_FILE``
pointed into a temporary directory. No provider, API key, or network is
involved: the ``session`` subcommand is purely local.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from src.main import _parse_args, main
from src.session import (
    DEFAULT_MAX_MESSAGES,
    SESSION_FILE_ENV_VAR,
    SESSION_VERSION,
    SessionStore,
)


class SessionCliTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.session_path = os.path.join(tmp.name, "session.json")
        env = mock.patch.dict(
            os.environ, {SESSION_FILE_ENV_VAR: self.session_path}
        )
        env.start()
        self.addCleanup(env.stop)

    def _run_cli(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()


class SessionStatusTest(SessionCliTestCase):
    def test_status_with_no_session(self):
        code, output = self._run_cli(["session", "status"])
        self.assertEqual(code, 0)
        self.assertIn(self.session_path, output)
        self.assertIn("no session yet", output)

    def test_status_reports_existing_session(self):
        store = SessionStore()
        store.save(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        code, output = self._run_cli(["session", "status"])
        self.assertEqual(code, 0)
        self.assertIn("Messages stored: 2", output)
        self.assertIn(f"version {SESSION_VERSION}", output)
        self.assertIn(str(DEFAULT_MAX_MESSAGES), output)

    def test_status_never_prints_message_content(self):
        store = SessionStore()
        store.save([{"role": "user", "content": "SECRET-TOKEN-abc123"}])
        _, output = self._run_cli(["session", "status"])
        self.assertNotIn("SECRET-TOKEN-abc123", output)

    def test_status_reports_corrupt_file_without_crashing(self):
        os.makedirs(os.path.dirname(self.session_path), exist_ok=True)
        with open(self.session_path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        code, output = self._run_cli(["session", "status"])
        self.assertEqual(code, 0)
        self.assertIn("unreadable/corrupt", output)


class SessionResetTest(SessionCliTestCase):
    def test_reset_clears_existing_session(self):
        store = SessionStore()
        store.save([{"role": "user", "content": "hi"}])
        self.assertTrue(os.path.exists(self.session_path))
        code, output = self._run_cli(["session", "reset"])
        self.assertEqual(code, 0)
        self.assertIn("Session cleared", output)
        self.assertFalse(os.path.exists(self.session_path))

    def test_reset_without_session_is_clean(self):
        code, output = self._run_cli(["session", "reset"])
        self.assertEqual(code, 0)
        self.assertIn("No session to clear", output)

    def test_next_start_is_fresh_after_reset(self):
        store = SessionStore()
        store.save([{"role": "user", "content": "old conversation"}])
        self._run_cli(["session", "reset"])
        self.assertFalse(SessionStore().status()["exists"])


class SessionCliArgumentTest(unittest.TestCase):
    def test_fresh_flag_parses(self):
        self.assertTrue(_parse_args(["--fresh"]).fresh_session)
        self.assertTrue(_parse_args(["--fresh-session"]).fresh_session)

    def test_fresh_flag_defaults_off(self):
        self.assertFalse(_parse_args([]).fresh_session)

    def test_session_subcommand_parses(self):
        args = _parse_args(["session", "status"])
        self.assertEqual(args.command, "session")
        self.assertEqual(args.session_action, "status")
        self.assertEqual(_parse_args(["session", "reset"]).session_action,
                         "reset")


if __name__ == "__main__":
    unittest.main()