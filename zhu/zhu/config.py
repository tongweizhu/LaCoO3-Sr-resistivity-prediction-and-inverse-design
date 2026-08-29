"""Runtime configuration shared by the Rio pages."""

from __future__ import annotations

import os


DEFAULT_BACKEND_URL = "http://127.0.0.1:5050"
BACKEND_URL = os.environ.get("BACKEND_URL", DEFAULT_BACKEND_URL).strip().rstrip("/")
if not BACKEND_URL:
    BACKEND_URL = DEFAULT_BACKEND_URL
