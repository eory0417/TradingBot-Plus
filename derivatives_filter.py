"""선물 펀딩비·미결제약정(OI) 기반 진입 필터."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from config import Settings, settings as default_settings

Side = Literal["long", "short"]


@dataclass(frozen=True, slots=True)
class DerivativesSnapshot:
    funding_rate_pct: float  # 예: 0.05 = 0.05%
    open_interest: float
    oi_change_pct: float
    ready: bool
    details: str = ""


def gate_entry(
    side: Side,
    snap: DerivativesSnapshot,
    cfg: Settings | None = None,
) -> tuple[bool, float, str]:
    """(허용, 사이즈배수, 사유). funding_rate_pct는 % 단위."""
    cfg = cfg or default_settings
    if not cfg.deriv_filter_enabled:
        return True, 1.0, "deriv off"
    if not snap.ready:
        if cfg.deriv_require_ready:
            return False, 0.0, "deriv not ready"
        return True, 1.0, "deriv not ready (pass)"

    fr = snap.funding_rate_pct
    mode = cfg.deriv_mode
    reduce_mult = float(cfg.deriv_reduce_mult)

    long_overheated = fr >= cfg.funding_long_block_pct
    short_overheated = fr <= cfg.funding_short_block_pct
    oi_spike = (
        cfg.oi_spike_pct > 0
        and snap.oi_change_pct >= cfg.oi_spike_pct
    )

    if side == "long" and long_overheated:
        extra = f" (OI+{snap.oi_change_pct:.1f}%)" if oi_spike else ""
        if mode == "block":
            return False, 0.0, f"funding long overheat {fr:.3f}%{extra}"
        return True, reduce_mult, f"funding long reduce {fr:.3f}%{extra}"

    if side == "short" and short_overheated:
        extra = f" (OI+{snap.oi_change_pct:.1f}%)" if oi_spike else ""
        if mode == "block":
            return False, 0.0, f"funding short overheat {fr:.3f}%{extra}"
        return True, reduce_mult, f"funding short reduce {fr:.3f}%{extra}"

    return True, 1.0, f"funding ok {fr:.3f}%"
