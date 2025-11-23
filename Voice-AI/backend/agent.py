"""Top-level test shim: re-export symbols from the package-style `src` layout.

Tests in this project import `agent` directly. The real implementation lives
in `src/agent.py` (package layout). This module re-exports the commonly used
symbols so tests can `from agent import Assistant` without installing the
package.
"""
from __future__ import annotations

from src.agent import Assistant, entrypoint, prewarm  # re-export for tests/CLI

__all__ = ["Assistant", "entrypoint", "prewarm"]
