"""뉴스 카테고리 분류 및 카테고리별 청산 프로필."""

from __future__ import annotations

import json
import re
from typing import Any

from config import Settings, settings as default_settings

CATEGORIES = (
    "listing",
    "hack",
    "mainnet",
    "partnership",
    "etf",
    "regulation",
    "bb_breakout",
    "default",
)

# 우선순위 높은 순 (먼저 매칭)
_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("listing", (
        "list on binance", "binance lists", "listing", "상장", "거래 지원", "거래지원",
        "will list", "to list",
    )),
    ("hack", (
        "hack", "exploit", "drain", "breach", "stolen", "해킹", "피싱", "exploited",
    )),
    ("etf", ("etf", "spot etf", "bitcoin etf", "eth etf")),
    ("mainnet", ("mainnet", "메인넷", "launch mainnet", "goes live")),
    ("regulation", (
        "sec", "regulation", "regulatory", "ban", "lawsuit", "규제", "소송", "금지",
    )),
    ("partnership", (
        "partnership", "partner", "collaborat", "integrat", "제휴", "파트너", "협력",
    )),
]

_DEFAULT_PROFILES: dict[str, dict[str, float | bool]] = {
    "listing": {
        "trailing_atr_mult": 3.0,
        "time_exit_hours": 8.0,
        "scale_out_fraction": 0.3,
        "trailing_profit_pct": 1.5,
    },
    "partnership": {
        "time_exit_hours": 0.05,
        "trailing_profit_pct": 0.5,
        "trailing_atr_mult": 1.5,
    },
    "hack": {
        "time_exit_hours": 1.0,
        "trailing_atr_mult": 1.2,
        "stop_loss_pct": 1.5,
    },
    "etf": {
        "trailing_atr_mult": 2.5,
        "time_exit_hours": 6.0,
    },
    "mainnet": {
        "trailing_atr_mult": 2.5,
        "time_exit_hours": 5.0,
    },
    "regulation": {
        "time_exit_hours": 2.0,
        "trailing_atr_mult": 1.5,
    },
    "bb_breakout": {},
    "default": {},
}


def _parse_profiles(raw: str) -> dict[str, dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return dict(_DEFAULT_PROFILES)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return dict(_DEFAULT_PROFILES)
    if not isinstance(data, dict):
        return dict(_DEFAULT_PROFILES)
    merged = dict(_DEFAULT_PROFILES)
    for k, v in data.items():
        if isinstance(v, dict):
            base = dict(merged.get(k, {}))
            base.update(v)
            merged[k] = base
    return merged


def classify_news(title: str, title_ko: str = "") -> str:
    """키워드 규칙으로 뉴스 카테고리 분류."""
    text = f"{title} {title_ko}".lower()
    text = re.sub(r"\s+", " ", text)
    for cat, keywords in _CATEGORY_RULES:
        for kw in keywords:
            if kw.lower() in text:
                return cat
    return "default"


_PROFILE_TO_POSITION: dict[str, str] = {
    "trailing_atr_mult": "atr_mult_base",
    "trailing_atr_mult_tight": "atr_mult_tight",
    "trailing_profit_pct": "trailing_profit_pct",
    "time_exit_hours": "time_exit_hours",
    "stop_loss_pct": "stop_loss_pct",
    "stop_loss_atr_mult": "stop_loss_atr_mult",
    "scale_out_fraction": "scale_out_fraction",
    "scale_out_atr_mult": "scale_out_atr_mult",
    "scale_out_enabled": "scale_out_enabled",
    "scale_out_move_be": "scale_out_move_be",
}


def category_exit_overrides(category: str, cfg: Settings | None = None) -> dict[str, Any]:
    """Position 생성 시 덮어쓸 청산 파라미터 dict."""
    if not (cfg or default_settings).news_category_tp_enabled:
        return {}
    profiles = _parse_profiles((cfg or default_settings).news_category_exit_profiles)
    prof = profiles.get(category) or profiles.get("default") or {}
    out: dict[str, Any] = {}
    for k, v in prof.items():
        pos_key = _PROFILE_TO_POSITION.get(k)
        if pos_key:
            out[pos_key] = v
    return out
