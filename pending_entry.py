"""하이브리드 진입 — IOC 1차 + Maker 눌림목 2차 대기."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from config import settings

Side = Literal["long", "short"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def compute_pullback_limit(
    side: Side,
    spike_high: float,
    spike_low: float,
    *,
    pullback_bps: float | None = None,
) -> float:
    bps = float(pullback_bps if pullback_bps is not None else settings.hybrid_pullback_bps)
    if side == "long":
        return spike_high * (1.0 - bps / 10_000.0)
    return spike_low * (1.0 + bps / 10_000.0)


@dataclass(slots=True)
class PendingEntry:
    symbol: str
    side: Side
    leverage: int
    maker_notional: float
    limit_price: float
    expires_at: datetime
    order_id: str | None = None
    news: str = ""
    news_ko: str = ""
    score: float = 0.0
    category: str = "default"
    atr: float = 0.0
    spike_high: float = 0.0
    spike_low: float = 0.0

    def expired(self, now: datetime | None = None) -> bool:
        ts = _now() if now is None else now
        return ts >= self.expires_at

    @classmethod
    def from_spike(
        cls,
        *,
        symbol: str,
        side: Side,
        leverage: int,
        maker_notional: float,
        spike_high: float,
        spike_low: float,
        news: str = "",
        news_ko: str = "",
        score: float = 0.0,
        category: str = "default",
        atr: float = 0.0,
        order_id: str | None = None,
    ) -> PendingEntry:
        wait = max(10, int(settings.hybrid_maker_wait_sec))
        limit = compute_pullback_limit(side, spike_high, spike_low)
        return cls(
            symbol=symbol,
            side=side,
            leverage=leverage,
            maker_notional=maker_notional,
            limit_price=limit,
            expires_at=_now().replace(microsecond=0) + timedelta(seconds=wait),
            order_id=order_id,
            news=news,
            news_ko=news_ko,
            score=score,
            category=category,
            atr=atr,
            spike_high=spike_high,
            spike_low=spike_low,
        )
