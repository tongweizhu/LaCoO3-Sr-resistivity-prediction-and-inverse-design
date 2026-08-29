"""Regression checks for the browser-level fixed workbench boundary."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "zhu"))

import zhu  # noqa: E402


class ViewportLockTest(unittest.IsolatedAsyncioTestCase):
    async def test_every_new_session_installs_a_no_scroll_viewport_lock(self) -> None:
        session = AsyncMock()

        await zhu._lock_viewport(session)

        session._evaluate_javascript.assert_awaited_once()
        script = session._evaluate_javascript.await_args.args[0]
        self.assertIn("lsco-fixed-viewport", script)
        self.assertIn("overflow: hidden !important", script)
        self.assertIn("rio-fundamental-root-component", script)
        self.assertIs(zhu.app._on_session_start, zhu._lock_viewport)


if __name__ == "__main__":
    unittest.main()
