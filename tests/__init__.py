"""Shared test bootstrap for local `src/` imports."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())
