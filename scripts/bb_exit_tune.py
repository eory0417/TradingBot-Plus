"""BB 진입 고정 + 청산 파라미터 탐색 (90일, 4코인 합산 수익)."""

from __future__ import annotations

import itertools
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.engine import BacktestCosts
from bb_breakout import BB_OHLCV_LIMIT, evaluate_bb_entry
from config import settings as app_settings
from scripts.bb_entry_tune import DAYS, SYMBOLS, SymbolData, _load_symbol
from strategy import Position

# BB 진입 고정 (사용자 첨부)
BB_FIXED = dict(
    bb_len=20,
    bb_mult=3.2,
    bb_min=1.1,
    vol_mult=1.5,
    vol_len=15,
    min_range_pct=0.05,
    bb_trend_mode="off",
    f_trend_len=0,
    bb_leverage=1,
    bb_max_add_leverage=1,
    position_size_usdt=100.0,
)

# 현재 기본 청산 (비교 기준)
BASELINE_EXIT = dict(
    stop_loss_mode="fixed",
    stop_loss_pct=2.0,
    stop_loss_atr_mult=2.0,
    trailing_profit_pct=2.0,
    trailing_atr_mult=2.0,
    trailing_atr_mult_tight=1.5,
    time_exit_hours=5.0,
)

COSTS = BacktestCosts(fee_pct=0.04, slippage_pct=0.02)
INITIAL = 10_000.0
CACHE_DIR = Path(__file__).resolve().parent / "cache"

EXIT_GRID = {
    "stop_loss_mode": ["fixed", "atr"],
    "sl": [1.5, 2.0, 2.5, 3.0],  # fixed=% / atr=ATR배수
    "trailing_profit_pct": [1.5, 2.0, 2.5, 3.0],
    "trailing_atr_mult": [1.5, 2.0, 2.5, 3.0],
    "time_exit_hours": [3.0, 5.0, 8.0],
}


@dataclass
class SimResult:
    net_pnl: float
    return_pct: float
    trades: int
    win_rate: float
    profit_factor: float
    max_dd_pct: float


def _cfg(exit_params: dict):
    c = app_settings.model_copy(deep=True, update={**BB_FIXED, **exit_params})
    return c


def _fee(notional: float) -> float:
    return notional * COSTS.fee_pct / 100


def _slip(price: float, side: str, *, entry: bool) -> float:
    if COSTS.slippage_pct <= 0:
        return price
    adj = price * COSTS.slippage_pct / 100
    if entry:
        return price + adj if side == "long" else price - adj
    return price - adj if side == "long" else price + adj


def simulate_symbol(data: SymbolData, cfg) -> SimResult:
    rows = data.rows_1m
    n = len(rows)
    free = INITIAL
    position: Position | None = None
    last_entry_bar: int | None = None
    trades_pnl: list[float] = []
    wins = 0
    equity: list[float] = []

    for i in range(BB_OHLCV_LIMIT - 1, n):
        if not data.trade_mask[i]:
            continue
        row = rows[i]
        ts = int(row[0])
        close = float(row[4])
        ind = data.ind_at[i]

        if position is not None:
            bar_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            sig = position.update(
                close,
                atr=ind.atr if ind else None,
                slope=ind.slope if ind else None,
                rsi=ind.rsi if ind else None,
                news_score=None,
                now=bar_time,
            )
            if sig.should_exit:
                exit_px = _slip(close, position.side, entry=False)
                pnl_pct = position.unrealized_pct()
                gross = position.margin * (pnl_pct / 100) * position.leverage
                exit_fee = _fee(position.notional)
                net = gross - exit_fee
                free += position.margin + net
                trades_pnl.append(net)
                if net > 0:
                    wins += 1
                position = None

        window = rows[i - BB_OHLCV_LIMIT + 1 : i + 1]
        entry = evaluate_bb_entry(window, cfg)
        if entry.ok and entry.side and position is None and ind is not None:
            if last_entry_bar != ts:
                margin = float(cfg.position_size_usdt)
                if free >= margin + _fee(float(cfg.position_size_usdt)):
                    entry_px = _slip(close, entry.side, entry=True)
                    entry_fee = _fee(float(cfg.position_size_usdt))
                    free -= margin + entry_fee
                    position = Position(
                        symbol="",
                        side=entry.side,
                        amount=float(cfg.position_size_usdt) / entry_px,
                        entry_price=entry_px,
                        atr=ind.atr if ind.atr > 0 else entry_px * 0.01,
                        opened_at=datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                        notional=float(cfg.position_size_usdt),
                        leverage=max(1, int(cfg.bb_leverage)),
                        margin=margin,
                        stop_loss_mode=cfg.stop_loss_mode,
                        stop_loss_pct=cfg.stop_loss_pct,
                        stop_loss_atr_mult=cfg.stop_loss_atr_mult,
                        atr_mult_base=cfg.trailing_atr_mult,
                        atr_mult_tight=cfg.trailing_atr_mult_tight,
                        trailing_profit_pct=cfg.trailing_profit_pct,
                        time_exit_hours=cfg.time_exit_hours,
                    )
                    last_entry_bar = ts

        eq = free
        if position is not None:
            position.mark_price = close
            unreal = position.margin * (position.unrealized_pct() / 100) * position.leverage
            eq = free + position.margin + unreal
        equity.append(eq)

    final = equity[-1] if equity else free
    gross_profit = sum(p for p in trades_pnl if p > 0)
    gross_loss = abs(sum(p for p in trades_pnl if p <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    peak = equity[0] if equity else final
    max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            max_dd = min(max_dd, (e - peak) / peak * 100)
    return SimResult(
        net_pnl=final - INITIAL,
        return_pct=(final / INITIAL - 1) * 100,
        trades=len(trades_pnl),
        win_rate=wins / len(trades_pnl) * 100 if trades_pnl else 0.0,
        profit_factor=pf,
        max_dd_pct=max_dd,
    )


def _exit_dict(mode: str, sl: float, tp: float, tatr: float, hours: float) -> dict:
    d = dict(
        stop_loss_mode=mode,
        trailing_profit_pct=tp,
        trailing_atr_mult=tatr,
        trailing_atr_mult_tight=1.5,
        time_exit_hours=hours,
        stop_loss_pct=2.0,
        stop_loss_atr_mult=2.0,
    )
    if mode == "fixed":
        d["stop_loss_pct"] = sl
    else:
        d["stop_loss_atr_mult"] = sl
    return d


def run_combo(cache: dict[str, SymbolData], exit_params: dict) -> dict:
    cfg = _cfg(exit_params)
    total_net = 0.0
    total_trades = 0
    per: dict[str, SimResult] = {}
    for sym, data in cache.items():
        r = simulate_symbol(data, cfg)
        per[sym] = r
        total_net += r.net_pnl
        total_trades += r.trades
    return {
        **exit_params,
        "total_net": total_net,
        "total_trades": total_trades,
        "per": per,
    }


def main() -> None:
    end = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    start = end - __import__("datetime").timedelta(days=DAYS)
    since = int(start.timestamp() * 1000)
    until = int(end.timestamp() * 1000)

    print(f"=== BB exit tune ({DAYS}d, BB fixed, 4 symbols x {INITIAL:.0f} USDT) ===\n")
    cache: dict[str, SymbolData] = {}
    for sym in SYMBOLS:
        cache[sym] = _load_symbol(sym, since, until)

    baseline = run_combo(cache, BASELINE_EXIT)
    print("BASELINE (current):")
    print(
        f"  fixed SL {BASELINE_EXIT['stop_loss_pct']}% | "
        f"trail {BASELINE_EXIT['trailing_profit_pct']}% / ATRx{BASELINE_EXIT['trailing_atr_mult']} | "
        f"time {BASELINE_EXIT['time_exit_hours']}h"
    )
    print(f"  total net: {baseline['total_net']:+.2f} USDT | trades: {baseline['total_trades']}")
    for sym in SYMBOLS:
        r = baseline["per"][sym]
        print(f"    {sym}: {r.net_pnl:+.2f} USDT ({r.trades} trades, WR {r.win_rate:.1f}%)")
    print()

    combos = list(
        itertools.product(
            EXIT_GRID["stop_loss_mode"],
            EXIT_GRID["sl"],
            EXIT_GRID["trailing_profit_pct"],
            EXIT_GRID["trailing_atr_mult"],
            EXIT_GRID["time_exit_hours"],
        )
    )
    print(f"Grid: {len(combos)} exit combos\n")
    t0 = time.perf_counter()
    scored: list[dict] = []
    for i, (mode, sl, tp, tatr, hours) in enumerate(combos):
        ep = _exit_dict(mode, sl, tp, tatr, hours)
        row = run_combo(cache, ep)
        row["improvement"] = row["total_net"] - baseline["total_net"]
        scored.append(row)
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(combos)} ({time.perf_counter() - t0:.1f}s)", flush=True)

    scored.sort(key=lambda x: x["total_net"], reverse=True)
    best = scored[0]
    print(f"\nDone in {time.perf_counter() - t0:.1f}s\n")
    print("=== BEST (total net PnL) ===")
    print(
        f"  손절: {best['stop_loss_mode']} "
        f"{'%='+str(best['stop_loss_pct']) if best['stop_loss_mode']=='fixed' else 'ATRx'+str(best['stop_loss_atr_mult'])}"
    )
    print(f"  Trail 이익 %: {best['trailing_profit_pct']}")
    print(f"  Trail ATR: {best['trailing_atr_mult']}")
    print(f"  시간청산: {best['time_exit_hours']}h")
    print(f"  total net: {best['total_net']:+.2f} USDT (vs baseline {best['improvement']:+.2f})")
    print(f"  trades: {best['total_trades']}")
    for sym in SYMBOLS:
        r = best["per"][sym]
        print(f"    {sym}: {r.net_pnl:+.2f} USDT | PF {r.profit_factor:.2f} | MDD {r.max_dd_pct:.1f}%")

    print("\n=== TOP 10 ===")
    for row in scored[:10]:
        sl_label = (
            f"SL {row['stop_loss_pct']}%"
            if row["stop_loss_mode"] == "fixed"
            else f"SL ATRx{row['stop_loss_atr_mult']}"
        )
        print(
            f"  {sl_label} | trail {row['trailing_profit_pct']}%/ATRx{row['trailing_atr_mult']} | "
            f"time {row['time_exit_hours']}h -> net {row['total_net']:+.2f} ({row['improvement']:+.2f})"
        )


if __name__ == "__main__":
    main()
