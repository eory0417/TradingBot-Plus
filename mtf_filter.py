"""다중 타임프레임(MTF) EMA 추세 필터.

1h/4h 등 상위 TF의 EMA와 종가 비교로 bull/bear/neutral bias를 산출하고,
진입 허용·사이즈 축소 배수를 결정한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from config import Settings, settings as default_settings

Bias = Literal["bull", "bear", "neutral"]
Side = Literal["long", "short"]


@dataclass(frozen=True, slots=True)
class MtfTrend:
    bias: Bias
    ready: bool
    details: str = ""


def ema_last(closes: np.ndarray | list[float], length: int) -> float | None:
    """종가 시리즈의 EMA(length) 마지막 값. 데이터 부족 시 None."""
    arr = np.asarray(closes, dtype=float)
    if arr.size < max(5, length // 4) or length < 2:
        return None
    s = pd.Series(arr)
    out = s.ewm(span=length, adjust=False).mean()
    val = float(out.iloc[-1])
    return val if np.isfinite(val) else None


def bias_from_close_ema(close: float, ema: float, *, band_pct: float = 0.0) -> Bias:
    if ema <= 0 or close <= 0:
        return "neutral"
    if band_pct > 0:
        upper = ema * (1 + band_pct / 100)
        lower = ema * (1 - band_pct / 100)
        if close > upper:
            return "bull"
        if close < lower:
            return "bear"
        return "neutral"
    if close > ema:
        return "bull"
    if close < ema:
        return "bear"
    return "neutral"


def combine_biases(biases: list[Bias]) -> Bias:
    """모든 TF가 동의할 때만 bull/bear, 아니면 neutral."""
    if not biases:
        return "neutral"
    if all(b == "bull" for b in biases):
        return "bull"
    if all(b == "bear" for b in biases):
        return "bear"
    return "neutral"


def evaluate_mtf_from_frames(
    frames: dict[str, pd.DataFrame],
    *,
    ema_len: int = 200,
    fallback_len: int = 50,
    band_pct: float = 0.0,
) -> MtfTrend:
    """timeframe -> OHLCV DataFrame(columns include close) 맵으로 MTF bias 계산."""
    if not frames:
        return MtfTrend(bias="neutral", ready=False, details="no frames")

    biases: list[Bias] = []
    parts: list[str] = []
    any_ready = False
    for tf, df in frames.items():
        if df is None or df.empty or "close" not in df.columns:
            parts.append(f"{tf}:empty")
            continue
        closes = df["close"].to_numpy(dtype=float)
        close = float(closes[-1])
        use_len = ema_len
        ema = ema_last(closes, ema_len)
        ready = ema is not None and closes.size >= ema_len
        if ema is None:
            ema = ema_last(closes, fallback_len)
            use_len = fallback_len
            ready = False
        if ema is None:
            parts.append(f"{tf}:no_ema")
            continue
        b = bias_from_close_ema(close, ema, band_pct=band_pct)
        biases.append(b)
        any_ready = any_ready or ready
        parts.append(f"{tf}:EMA{use_len}={ema:.4f} close={close:.4f}→{b}")

    if not biases:
        return MtfTrend(bias="neutral", ready=False, details="; ".join(parts) or "no bias")
    return MtfTrend(
        bias=combine_biases(biases),
        ready=any_ready and len(biases) == len(frames),
        details="; ".join(parts),
    )


def mtf_allows_side(bias: Bias, side: Side, *, mode: str) -> bool:
    """block 모드에서 역추세 진입 차단. reduce/off는 항상 허용."""
    if mode != "block":
        return True
    if bias == "neutral":
        return True
    if side == "long" and bias == "bear":
        return False
    if side == "short" and bias == "bull":
        return False
    return True


def mtf_size_mult(bias: Bias, side: Side, *, mode: str, reduce_mult: float) -> float:
    """reduce 모드에서 역추세 시 사이즈 배수. 그 외 1.0."""
    if mode != "reduce":
        return 1.0
    against = (side == "long" and bias == "bear") or (side == "short" and bias == "bull")
    if against:
        return max(0.0, float(reduce_mult))
    return 1.0


def parse_mtf_tfs(raw: str) -> list[str]:
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    return parts or ["1h", "4h"]


def gate_entry(
    side: Side,
    trend: MtfTrend,
    cfg: Settings | None = None,
) -> tuple[bool, float, str]:
    """(허용여부, 사이즈배수, 사유). 비활성/미준비면 허용·배수1."""
    cfg = cfg or default_settings
    if not cfg.mtf_filter_enabled:
        return True, 1.0, "mtf off"
    if not trend.ready and cfg.mtf_require_ready:
        return False, 0.0, f"mtf not ready ({trend.details})"
    mode = cfg.mtf_mode
    if not mtf_allows_side(trend.bias, side, mode=mode):
        return False, 0.0, f"mtf block {side} vs {trend.bias} ({trend.details})"
    mult = mtf_size_mult(trend.bias, side, mode=mode, reduce_mult=cfg.mtf_reduce_mult)
    if mult <= 0:
        return False, 0.0, f"mtf reduce→0 {side} vs {trend.bias}"
    return True, mult, f"mtf {trend.bias} x{mult:g}"
