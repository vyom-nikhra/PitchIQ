"""Minimal .env loader — keeps secrets out of code, configs and commits.

Reads ``<repo>/.env`` (gitignored) into ``os.environ`` at import time without
overriding variables already set in the real environment. No python-dotenv
dependency; the format is strictly ``KEY=value`` lines with ``#`` comments.
Secrets must only ever be accessed via :func:`get_secret` so there is exactly
one audited path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_LOADED = False


def load_env(path: str | Path | None = None) -> None:
    global _LOADED
    if _LOADED and path is None:
        return
    env_path = Path(path) if path else Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    _LOADED = True


def get_secret(name: str) -> str | None:
    """Fetch a secret from env (after .env load). Never log the value."""
    load_env()
    value = os.environ.get(name)
    if value:
        log.debug("secret %s: present", name)
    return value or None
