"""변동성·반켈리 기반 포지션 사이징."""

from __future__ import annotations

from dataclasses import dataclass, field

from config import Settings, settings as default_settings


@dataclass
class TradeStats:
    """최근 거래 승률·손익 통계 (반켈리용)."""

    wins: int = 0
    losses: int = 0
    sum_win_pct: float = 0.0
    sum_loss_pct: float = 0.0  # 양수 합(절대값)

    def record(self, pnl_pct: float) -> None:
        if pnl_pct > 0:
            self.wins += 1
            self.sum_win_pct += pnl_pct
        else:
            self.losses += 1
            self.sum_loss_pct += abs(pnl_pct)

    @property
    def n(self) -> int:
        return self.wins + self.losses

    def half_kelly_fraction(self, *, max_fraction: float) -> float:
        """반켈리 f*/2. 데이터 부족 시 1.0(캡만 적용하지 않음 → 사이즈 배수 1)."""
        if self.n < 10 or self.wins == 0 or self.losses == 0:
            return 1.0
        p = self.wins / self.n
        q = 1.0 - p
        avg_w = self.sum_win_pct / self.wins
        avg_l = self.sum_loss_pct / self.losses
        if avg_l <= 0 or avg_w <= 0:
            return 1.0
        b = avg_w / avg_l
        f = (b * p - q) / b
        half = max(0.0, f) * 0.5
        # half kelly를 "배수"로 쓰되 1.0을 기준으로 cap
        if half <= 0:
            return 0.0
        # 해석: 권장 베팅 비율이 작으면 사이즈 축소. max_fraction은 상한 배수.
        return min(max_fraction, max(0.05, half * 10.0))  # scale small f to ~size mult


@dataclass
class SizingState:
    stats: TradeStats = field(default_factory=TradeStats)


def atr_pct(atr: float, price: float) -> float:
    if price <= 0 or atr <= 0:
        return 0.0
    return (atr / price) * 100.0


def vol_scalar(
    current_atr_pct: float,
    *,
    target_atr_pct: float,
    min_mult: float,
    max_mult: float,
) -> float:
    if current_atr_pct <= 0 or target_atr_pct <= 0:
        return 1.0
    raw = target_atr_pct / current_atr_pct
    return float(min(max_mult, max(min_mult, raw)))


def compute_notional(
    base_usdt: float,
    *,
    atr: float,
    price: float,
    cfg: Settings | None = None,
    sizing_state: SizingState | None = None,
    regime_scalar: float = 1.0,
    extra_mult: float = 1.0,
) -> float:
    """최종 진입 명목(USDT)."""
    cfg = cfg or default_settings
    mode = (cfg.sizing_mode or "fixed").strip().lower()
    mult = max(0.0, float(extra_mult)) * max(0.0, float(regime_scalar))

    if mode in {"vol", "vol_kelly"}:
        cur = atr_pct(atr, price)
        mult *= vol_scalar(
            cur,
            target_atr_pct=cfg.vol_target_atr_pct,
            min_mult=cfg.sizing_min_mult,
            max_mult=cfg.sizing_max_mult,
        )
    if mode == "vol_kelly" and sizing_state is not None:
        k = sizing_state.stats.half_kelly_fraction(max_fraction=cfg.kelly_max_fraction)
        mult *= k

    notional = float(base_usdt) * mult
    # 하한: 너무 작으면 주문 실패 → 최소 1 USDT
    return max(1.0, notional)
