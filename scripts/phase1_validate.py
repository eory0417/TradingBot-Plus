"""Phase1 검증: 기준 vs MTF+Scale-out+Vol 사이징 (90일 4코인)."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.engine import BacktestCosts, fetch_backtest_frames, run_backtest
from config import settings as app_settings
from mtf_filter import parse_mtf_tfs
from scripts.bb_entry_tune import DAYS, SYMBOLS

COSTS = BacktestCosts(fee_pct=0.04, slippage_pct=0.02)
BB = dict(
    bb_len=20, bb_mult=3.2, bb_min=1.1, vol_mult=1.5, vol_len=15,
    min_range_pct=0.05, bb_trend_mode="off", bb_leverage=1, bb_max_add_leverage=1,
    position_size_usdt=100.0, stop_loss_mode="fixed", stop_loss_pct=2.0,
    trailing_profit_pct=2.0, trailing_atr_mult=2.0, time_exit_hours=3.0,
)


def run_case(name: str, overrides: dict, frames: dict, since: int, until: int) -> None:
    cfg = app_settings.model_copy(deep=True, update={**BB, **overrides})
    total = 0.0
    trades = 0
    print(f"\n=== {name} ===")
    for sym, (df_1m, df_ind, mtf) in frames.items():
        r = run_backtest(
            df_1m, df_ind, sym, cfg=cfg, initial_capital=10_000.0, costs=COSTS,
            trade_start_ms=since, trade_end_ms=until,
            mtf_frames=mtf if cfg.mtf_filter_enabled else {},
        )
        net = r.final_equity - r.initial_capital
        total += net
        trades += r.total_trades
        print(f"  {sym}: {net:+.2f} USDT | trades={r.total_trades} WR={r.win_rate:.1f}%")
    print(f"  TOTAL: {total:+.2f} USDT | trades={trades}")


def main() -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS)
    since = int(start.timestamp() * 1000)
    until = int(end.timestamp() * 1000)
    frames: dict = {}
    tfs = parse_mtf_tfs("1h,4h")
    for sym in SYMBOLS:
        print(f"fetch {sym}...", flush=True)
        df_1m, df_ind, mtf = fetch_backtest_frames(
            sym, since_ms=since, until_ms=until, indicator_tf="15m",
            testnet=False, mtf_tfs=tfs, mtf_ema_len=200,
        )
        frames[sym] = (df_1m, df_ind, mtf)
        print(f"  1m={len(df_1m)} ind={len(df_ind)} mtf={[(k, len(v)) for k, v in mtf.items()]}")

    run_case(
        "BASELINE (no MTF, no scale-out, fixed size)",
        dict(mtf_filter_enabled=False, scale_out_enabled=False, sizing_mode="fixed"),
        frames, since, until,
    )
    run_case(
        "SCALE-OUT only",
        dict(
            mtf_filter_enabled=False, scale_out_enabled=True, scale_out_fraction=0.4,
            scale_out_atr_mult=1.5, scale_out_move_be=True, sizing_mode="fixed",
        ),
        frames, since, until,
    )
    run_case(
        "MTF block + SCALE-OUT + VOL sizing",
        dict(
            mtf_filter_enabled=True, mtf_mode="block", scale_out_enabled=True,
            scale_out_fraction=0.4, scale_out_atr_mult=1.5, scale_out_move_be=True,
            sizing_mode="vol", vol_target_atr_pct=0.8,
        ),
        frames, since, until,
    )


if __name__ == "__main__":
    main()
