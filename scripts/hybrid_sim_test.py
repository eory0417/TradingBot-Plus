"""하이브리드 진입 SIM 통합 테스트 (LIVE 자격증명 없이 강제 페이퍼 모드)."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from trading_engine import compute_indicators_from_df
import pandas as pd

# SIM 강제 + 하이브리드 ON
settings.hybrid_entry_enabled = True
settings.hybrid_ioc_fraction = 0.3
settings.hybrid_pullback_bps = 15.0
settings.hybrid_maker_wait_sec = 300
settings.mtf_filter_enabled = False
settings.news_entry_grace_seconds = 0


async def main() -> None:
    import bot as botmod
    from bot import TradingBot

    # 실제 API 키가 있어도 이 스크립트만 SIM으로 동작
    botmod.has_real_credentials = lambda: False  # type: ignore[method-assign]

    b = TradingBot()
    assert b.sim and b.sim_market is not None, "SIM 모드 아님"

    symbol = "BTC/USDT"
    ohlcv = b.sim_market.ohlcv(symbol, limit=120)
    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    ind = compute_indicators_from_df(symbol, settings.timeframe, df)

    print("=== 1) 하이브리드 IOC 진입 ===")
    await b._open(
        symbol, "long", ind,
        news="Binance lists TEST coin partnership",
        score=0.85, leverage=2,
        news_triggered_at=datetime.now(timezone.utc),
        use_hybrid=True, entry_category="listing",
    )

    pos = b.positions.get(symbol)
    pending = b._pending_entries.get(symbol)
    if pos is None:
        print("FAIL: 포지션 없음")
        sys.exit(1)
    if pending is None:
        print("FAIL: pending entry 없음")
        sys.exit(1)

    ioc_notional = pos.notional
    maker_notional = pending.maker_notional
    print(f"  IOC 체결: {ioc_notional:.2f} USDT (수량 {pos.amount:.6f} @ {pos.entry_price:.4f})")
    print(f"  Maker 대기: {maker_notional:.2f} USDT @ limit {pending.limit_price:.4f}")
    print(f"  카테고리: {pos.entry_category} | time_exit={pos.time_exit_hours}h")

    print("\n=== 2) 가격 눌림 → Maker 2차 체결 ===")
    # Long: mark <= limit 이면 체결
    target = pending.limit_price * 0.998
    b.sim_market._closes[symbol][-1] = target
    ind2 = await b._indicators(symbol)
    assert ind2 is not None
    await b._monitor_pending_entry(symbol, ind2)

    if symbol in b._pending_entries:
        print("FAIL: Maker 미체결 (pending 남음)")
        sys.exit(1)

    pos2 = b.positions[symbol]
    total = pos2.notional
    print(f"  합산 명목: {total:.2f} USDT (IOC {ioc_notional:.2f} + Maker ~{maker_notional:.2f})")
    print(f"  평균가: {pos2.entry_price:.4f} | 수량: {pos2.amount:.6f}")

    # 기대: 총 명목 ≈ IOC + Maker (±2%)
    expected = ioc_notional + maker_notional
    if abs(total - expected) / expected > 0.02:
        print(f"WARN: 명목 합 불일치 (기대 {expected:.2f}, 실제 {total:.2f})")
    else:
        print("  OK: IOC+Maker 합산 명목 일치")

    print("\n=== 3) 만료 시나리오 (별도 심볼) ===")
    sym2 = "ETH/USDT"
    ohlcv2 = b.sim_market.ohlcv(sym2, limit=120)
    df2 = pd.DataFrame(
        ohlcv2, columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    ind_e = compute_indicators_from_df(sym2, settings.timeframe, df2)
    settings.hybrid_maker_wait_sec = 0  # 즉시 만료
    await b._open(
        sym2, "short", ind_e,
        news="Hack exploit on bridge",
        score=-0.9, leverage=2,
        use_hybrid=True, entry_category="hack",
    )
    pending_e = b._pending_entries.get(sym2)
    if pending_e:
        # expires_at을 과거로
        from datetime import timedelta
        pending_e.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        ind_e2 = await b._indicators(sym2)
        await b._monitor_pending_entry(sym2, ind_e2)
        if sym2 not in b._pending_entries:
            print("  OK: 만료 시 pending 제거됨")
        else:
            print("FAIL: 만료 처리 실패")
            sys.exit(1)

    print("\n=== 하이브리드 SIM 테스트 통과 ===")


if __name__ == "__main__":
    asyncio.run(main())
