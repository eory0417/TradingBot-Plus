"""뉴스 소스·키워드 가중치 (규칙 기반 점수 후처리)."""

from __future__ import annotations

import json
import re
from typing import Any

from config import Settings, settings as default_settings


def _parse_json_map(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def source_weight(source: str, cfg: Settings | None = None) -> float:
    cfg = cfg or default_settings
    weights = _parse_json_map(cfg.news_source_weights)
    if not weights:
        return 1.0
    key = (source or "").strip().lower()
    if not key:
        return float(weights.get("default", 1.0))
    if key in weights:
        return float(weights[key])
    # 부분 매칭 (rss url host 등)
    for k, v in weights.items():
        if k != "default" and k.lower() in key:
            return float(v)
    return float(weights.get("default", 1.0))


def keyword_boost(symbol: str, text: str, cfg: Settings | None = None) -> float:
    """심볼별 키워드 배수. NEWS_KEYWORD_BOOST JSON:
    {"ETH": {"etf": 1.3, "upgrade": 1.2}, "default": {"hack": 0.5}}
    """
    cfg = cfg or default_settings
    table = _parse_json_map(cfg.news_keyword_boost)
    if not table or not text:
        return 1.0
    sym = (symbol or "").upper().replace("/USDT", "").replace(":USDT", "")
    buckets: list[dict] = []
    if sym in table and isinstance(table[sym], dict):
        buckets.append(table[sym])
    if "default" in table and isinstance(table["default"], dict):
        buckets.append(table["default"])
    text_l = text.lower()
    mult = 1.0
    for bucket in buckets:
        for kw, w in bucket.items():
            if str(kw).lower() in text_l:
                mult *= float(w)
    return mult


def effective_news_score(
    raw_score: float,
    *,
    source: str = "",
    symbol: str = "",
    title: str = "",
    title_ko: str = "",
    cfg: Settings | None = None,
) -> float:
    """threshold 비교 전 적용할 실효 점수. [-1, 1]로 클램프."""
    cfg = cfg or default_settings
    sw = source_weight(source, cfg)
    kb = keyword_boost(symbol, f"{title} {title_ko}", cfg)
    score = float(raw_score) * sw * kb
    return max(-1.0, min(1.0, score))


def parse_keyword_list(raw: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,|]", raw or "") if p.strip()]
