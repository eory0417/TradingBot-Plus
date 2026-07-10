"""청산 거래 영속 로그 (카테고리·뉴스 포함)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import BASE_DIR, settings

_lock = threading.Lock()
_PATH = BASE_DIR / settings.log_dir / "trades.jsonl"


def _ensure_path() -> Path:
    path = _PATH
    if not path.is_absolute():
        path = BASE_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def record_trade(trade: dict[str, Any]) -> None:
    """거래 한 건을 JSONL에 추가."""
    row = dict(trade)
    row.setdefault("logged_at", datetime.now(timezone.utc).isoformat())
    line = json.dumps(row, ensure_ascii=False)
    with _lock:
        path = _ensure_path()
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_trades(limit: int = 5000) -> list[dict[str, Any]]:
    path = _ensure_path()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with _lock:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows[-limit:]
