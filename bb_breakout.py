"""볼린저 밴드 돌파 진입 조건 (BBBQ_요구사항.md).

순수 함수만 제공 — 네트워크·주문 없음.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from config import Settings, settings

Side = Literal["long", "short"]
FailReason = Literal["BB_WIDTH", "VOLUME", "TREND", "RANGE"] | None

BB_OHLCV_LIMIT = 100


@dataclass(frozen=True, slots=True)
class BollingerBands:
    basis: float
    upper: float
    lower: float
    bb_width_pct: float


@dataclass(frozen=True, slots=True)
class BbEntryResult:
    ok: bool
    side: Side | None = None
    reason: FailReason = None
    volume_ratio: float | None = None


def compute_bollinger(
    closes: np.ndarray | list[float],
    bb_len: int,
    bb_mult: float,
) -> BollingerBands | None:
    """최근 bb_len 종가로 볼린저 밴드를 계산한다."""
    arr = np.asarray(closes, dtype=float)
    if arr.size < bb_len or bb_len < 1:
        return None
    window = arr[-bb_len:]
    basis = float(np.mean(window))
    if basis == 0:
        return None
    std = float(np.sqrt(np.mean((window - basis) ** 2)))
    upper = basis + bb_mult * std
    lower = basis - bb_mult * std
    bb_width_pct = (2 * bb_mult * std / basis) * 100
    return BollingerBands(basis=basis, upper=upper, lower=lower, bb_width_pct=bb_width_pct)


def _bb_width_ok(bb: BollingerBands, bb_min: float) -> bool:
    return bb_min == 0 or bb.bb_width_pct >= bb_min


def _raw_breakout_side(price: float, bb: BollingerBands) -> Side | None:
    """종가가 upper/lower를 넘었는지만 판정 (밴드 폭·필터 무관)."""
    long_ok = price > bb.upper
    short_ok = price < bb.lower
    if long_ok and short_ok:
        long_dist = price - bb.upper
        short_dist = bb.lower - price
        return "long" if long_dist >= short_dist else "short"
    if long_ok:
        return "long"
    if short_ok:
        return "short"
    return None


def _check_volume(volumes: np.ndarray, vol_len: int, vol_mult: float) -> tuple[bool, float]:
    if volumes.size < vol_len + 1:
        return False, 0.0
    volume = float(volumes[-1])
    prev = volumes[-(vol_len + 1) : -1]
    avg = float(np.mean(prev)) if prev.size else 0.0
    if avg == 0:
        return True, 0.0
    ratio = volume / avg
    return ratio >= vol_mult, ratio


def _check_trend(
    closes: np.ndarray,
    side: Side,
    f_trend_len: int,
    f_trend_pct: float,
    same_dir_min: float = 0.6,
) -> bool:
    if f_trend_len == 0:
        return True
    need = f_trend_len + 1
    if closes.size < need:
        return False
    window = closes[-need:]
    first = float(window[0])
    last = float(window[-1])
    if first == 0:
        return False
    slope_pct = ((last - first) / first) * 100
    if last > first:
        direction = "up"
    elif last < first:
        direction = "down"
    else:
        direction = "flat"

    deltas = np.diff(window)
    total = deltas.size
    if total == 0:
        return False
    if direction == "up":
        same = int(np.sum(deltas > 0))
    elif direction == "down":
        same = int(np.sum(deltas < 0))
    else:
        same = 0
    same_dir_ratio = same / total

    ok_dir = (side == "long" and direction == "up") or (side == "short" and direction == "down")
    ok_pct = abs(slope_pct) >= f_trend_pct
    ok_major = same_dir_ratio >= same_dir_min
    return ok_dir and ok_pct and ok_major


def _check_range(high: float, low: float, price: float, min_range_pct: float) -> bool:
    if min_range_pct == 0 or price == 0:
        return True
    range_pct = abs(high - low) / price * 100
    return range_pct >= min_range_pct


def evaluate_bb_entry(
    ohlcv_rows: list[list[float]] | np.ndarray,
    cfg: Settings | None = None,
) -> BbEntryResult:
    """1분봉 OHLCV로 BB 돌파 진입 가능 여부를 판정한다.

    ``ohlcv_rows``: ``(n, 6)`` ndarray 또는 ``[[ts, o, h, l, c, v], ...]``
    """
    cfg = cfg or settings
    arr = np.asarray(ohlcv_rows, dtype=float)
    if arr.shape[0] < BB_OHLCV_LIMIT:
        return BbEntryResult(ok=False, reason=None)

    closes = arr[:, 4]
    volumes = arr[:, 5]
    last = arr[-1]
    price = float(last[4])
    high = float(last[2])
    low = float(last[3])
    volume = float(last[5])

    bb = compute_bollinger(closes, cfg.bb_len, cfg.bb_mult)
    if bb is None:
        return BbEntryResult(ok=False, reason=None)

    side = _raw_breakout_side(price, bb)
    if side is None:
        return BbEntryResult(ok=False, reason=None)

    if not _bb_width_ok(bb, cfg.bb_min):
        return BbEntryResult(ok=False, side=side, reason="BB_WIDTH")

    vol_ok, vol_ratio = _check_volume(volumes, cfg.vol_len, cfg.vol_mult)
    if not vol_ok:
        return BbEntryResult(ok=False, side=side, reason="VOLUME", volume_ratio=vol_ratio)

    try:
        trend_cfg = cfg.bb_trend_filter()
        if trend_cfg is None:
            trend_ok = True
        else:
            t_len, t_pct, same_dir_min = trend_cfg
            trend_ok = _check_trend(closes, side, t_len, t_pct, same_dir_min)
    except Exception:  # noqa: BLE001 — fail-closed
        trend_ok = False
    if not trend_ok:
        return BbEntryResult(ok=False, side=side, reason="TREND", volume_ratio=vol_ratio)

    if not _check_range(high, low, price, cfg.min_range_pct):
        return BbEntryResult(ok=False, side=side, reason="RANGE", volume_ratio=vol_ratio)

    return BbEntryResult(ok=True, side=side, volume_ratio=vol_ratio)


def bb_series_for_chart(
    closes: list[float] | np.ndarray,
    bb_len: int,
    bb_mult: float,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """차트용 rolling BB 상·중·하단 시리즈 (초기 bb_len-1 구간은 None)."""
    arr = np.asarray(closes, dtype=float)
    upper: list[float | None] = []
    basis: list[float | None] = []
    lower: list[float | None] = []
    for i in range(arr.size):
        if i + 1 < bb_len:
            upper.append(None)
            basis.append(None)
            lower.append(None)
            continue
        bb = compute_bollinger(arr[: i + 1], bb_len, bb_mult)
        if bb is None:
            upper.append(None)
            basis.append(None)
            lower.append(None)
        else:
            upper.append(bb.upper)
            basis.append(bb.basis)
            lower.append(bb.lower)
    return upper, basis, lower
