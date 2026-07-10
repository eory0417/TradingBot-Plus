"""Fast Phase1 validation on BTC only (cached 1m if available via fresh fetch)."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.engine import BacktestCosts, fetch_backtest_frames, run_backtest
from config import settings as app_settings
from mtf_filter import parse_mtf_tfs

COSTS = BacktestCosts(fee_pct=0.04, slippage_pct=0.02)
BB = dict(
    bb_len=20, bb_mult=3.2, bb_min=1.1, vol_mult=1.5, vol_len=15,
    min_range_pct=0.05, bb_trend_mode="off", bb_leverage=1, bb_max_add_leverage=1,
    position_size_usdt=100.0, stop_loss_mode="fixed", stop_loss_pct=2.0,
    trailing_profit_pct=2.0, trailing_atr_mult=2.0, time_exit_hours=3.0,
)
SYMBOL = "BTC/USDT"
DAYS = 90


def main() -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS)
    since = int(start.timestamp() * 1000)
    until = int(end.timestamp() * 1000)
    tfs = parse_mtf_tfs("1h,4h")
    print(f"fetch {SYMBOL}...", flush=True)
    t0 = time.perf_counter()
    df_1m, df_ind, mtf = fetch_backtest_frames(
        SYMBOL, since_ms=since, until_ms=until, indicator_tf="15m",
        testnet=False, mtf_tfs=tfs, mtf_ema_len=200,
    )
    print(f"fetched in {time.perf_counter()-t0:.1f}s 1m={len(df_1m)}", flush=True)

    cases = [
        ("BASELINE", dict(mtf_filter_enabled=False, scale_out_enabled=False, sizing_mode="fixed")),
        ("SCALEOUT", dict(
            mtf_filter_enabled=False, scale_out_enabled=True, scale_out_fraction=0.4,
            scale_out_atr_mult=1.5, scale_out_move_be=True, sizing_mode="fixed",
        )),
        ("MTF+SCALE+VOL", dict(
            mtf_filter_enabled=True, mtf_mode="block", scale_out_enabled=True,
            scale_out_fraction=0.4, scale_out_atr_mult=1.5, scale_out_move_be=True,
            sizing_mode="vol", vol_target_atr_pct=0.8,
        )),
    ]
    for name, ov in cases:
        cfg = app_settings.model_copy(deep=True, update={**BB, **ov})
        t1 = time.perf_counter()
        r = run_backtest(
            df_1m, df_ind, SYMBOL, cfg=cfg, initial_capital=10_000.0, costs=COSTS,
            trade_start_ms=since, trade_end_ms=until,
            mtf_frames=mtf if cfg.mtf_filter_enabled else {},
        )
        net = r.final_equity - r.initial_capital
        print(
            f"{name}: net={net:+.2f} trades={r.total_trades} WR={r.win_rate:.1f}% "
            f"PF={r.profit_factor if r.profit_factor != float('inf') else 'inf'} "
            f"({time.perf_counter()-t1:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
