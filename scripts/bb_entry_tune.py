"""BB 진입 파라미터 탐색 — 90일·코인당 진입 횟수 목표 (청산 설정 고정)."""

from __future__ import annotations

import itertools
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.engine import BacktestCosts, fetch_backtest_frames, run_backtest
from bb_breakout import BB_OHLCV_LIMIT, evaluate_bb_entry
from config import settings as app_settings
from strategy import Position
from trading_engine import compute_indicators_from_df

# 청산·진입금 고정 (사용자 첨부)
FIXED = dict(
    stop_loss_mode="fixed",
    stop_loss_pct=2.0,
    trailing_profit_pct=2.0,
    trailing_atr_mult=2.0,
    time_exit_hours=5.0,
    position_size_usdt=100.0,
    bb_leverage=1,
    bb_max_add_leverage=1,
    bb_trend_mode="off",
    f_trend_len=0,
)

TARGET_ENTRIES = 40
DAYS = 90
SYMBOLS = ["ETH/USDT", "BTC/USDT", "SOL/USDT", "XRP/USDT"]
COSTS = BacktestCosts(fee_pct=0.04, slippage_pct=0.02)
CACHE_DIR = Path(__file__).resolve().parent / "cache"

# 1차: 기존 그리드 + 완화 구간
GRID = {
    "bb_len": [10, 15, 20],
    "bb_mult": [1.0, 1.2, 1.5],
    "bb_min": [0.0, 0.3],
    "vol_mult": [0.8, 1.0, 1.2],
    "vol_len": [10, 15],
    "min_range_pct": [0.0, 0.03],
}


@dataclass
class SymbolData:
    rows_1m: np.ndarray
    trade_mask: np.ndarray
    ind_at: list


def _base_cfg():
    return app_settings.model_copy(deep=True, update=FIXED)


def _build_ind_at(df_1m: np.ndarray, df_15m, symbol: str, tf: str) -> list:
    """각 1m 봉 시점에 대응하는 15m 지표를 미리 계산한다."""
    ind_ts = df_15m["timestamp"].to_numpy(dtype=np.int64)
    n = len(df_1m)
    out: list = [None] * n
    last_ind = None
    ind_idx = 0
    for i in range(n):
        ts = int(df_1m[i, 0])
        while ind_idx + 1 < len(ind_ts) and ind_ts[ind_idx + 1] <= ts:
            ind_idx += 1
            if ind_idx >= 19:
                sub = df_15m.iloc[: ind_idx + 1]
                last_ind = compute_indicators_from_df(symbol, tf, sub)
        out[i] = last_ind if ind_idx >= 19 else None
    return out


def _load_symbol(sym: str, since: int, until: int) -> SymbolData:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{sym.replace('/', '_')}_{DAYS}d.pkl"
    path = CACHE_DIR / key
    if path.exists():
        print(f"  cache hit {sym}", flush=True)
        with path.open("rb") as f:
            payload = pickle.load(f)
        return payload

        print(f"  fetching {sym}...", flush=True)
    df_1m, df_15m, _mtf = fetch_backtest_frames(
        sym, since_ms=since, until_ms=until, indicator_tf="15m", testnet=False,
    )
    rows = df_1m.to_numpy()
    ts = rows[:, 0].astype(np.int64)
    trade_mask = (ts >= since) & (ts <= until)
    print(f"    1m={len(rows):,} 15m={len(df_15m):,} - building 15m lookup...", flush=True)
    ind_at = _build_ind_at(rows, df_15m, sym, "15m")
    data = SymbolData(rows_1m=rows, trade_mask=trade_mask, ind_at=ind_at)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def count_entries_fast(data: SymbolData, cfg) -> int:
    """포지션·청산 시뮬레이션으로 실제 진입 횟수만 빠르게 집계."""
    rows = data.rows_1m
    n = len(rows)
    position: Position | None = None
    entered = 0
    last_entry_bar: int | None = None

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
                position = None

        window = rows[i - BB_OHLCV_LIMIT + 1 : i + 1]
        result = evaluate_bb_entry(window, cfg)
        if result.ok and result.side and position is None and ind is not None:
            if last_entry_bar != ts:
                entered += 1
                entry_px = close
                position = Position(
                    symbol="",
                    side=result.side,
                    amount=float(cfg.position_size_usdt) / entry_px,
                    entry_price=entry_px,
                    atr=ind.atr if ind.atr > 0 else entry_px * 0.01,
                    opened_at=datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                    notional=float(cfg.position_size_usdt),
                    leverage=max(1, int(cfg.bb_leverage)),
                    margin=float(cfg.position_size_usdt),
                    stop_loss_mode=cfg.stop_loss_mode,
                    stop_loss_pct=cfg.stop_loss_pct,
                    stop_loss_atr_mult=cfg.stop_loss_atr_mult,
                    atr_mult_base=cfg.trailing_atr_mult,
                    atr_mult_tight=cfg.trailing_atr_mult_tight,
                    trailing_profit_pct=cfg.trailing_profit_pct,
                    time_exit_hours=cfg.time_exit_hours,
                )
                last_entry_bar = ts

    return entered


def _verify_best(params: dict, cache: dict[str, SymbolData], since: int, until: int) -> dict[str, int]:
    """최종 후보는 엔진 run_backtest와 교차 검증."""
    cfg = _base_cfg()
    for k, v in params.items():
        setattr(cfg, k, v)
    out: dict[str, int] = {}
    for sym, data in cache.items():
        df_1m = __import__("pandas").DataFrame(
            data.rows_1m, columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        # 15m df는 검증 시 ind_missing 방지용 — fast path와 동일 ind_at 사용
        r = run_backtest(
            df_1m, __import__("pandas").DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
            sym, cfg=cfg, initial_capital=10_000.0, costs=COSTS,
            trade_start_ms=since, trade_end_ms=until,
        )
        out[sym] = r.entry_rejects.entered
    return out


def main() -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS)
    since = int(start.timestamp() * 1000)
    until = int(end.timestamp() * 1000)

    print(f"=== BB entry tune ({DAYS}d, target>={TARGET_ENTRIES}/coin) ===\n", flush=True)
    t0 = time.perf_counter()
    cache: dict[str, SymbolData] = {}
    for sym in SYMBOLS:
        cache[sym] = _load_symbol(sym, since, until)
    print(f"Data ready in {time.perf_counter() - t0:.1f}s\n", flush=True)

    keys = list(GRID.keys())
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"Grid: {len(combos)} combos × {len(SYMBOLS)} symbols\n", flush=True)

    hits: list[dict] = []
    scored: list[dict] = []
    t1 = time.perf_counter()

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        cfg = _base_cfg()
        for k, v in params.items():
            setattr(cfg, k, v)

        per_sym: dict[str, int] = {}
        for sym, data in cache.items():
            per_sym[sym] = count_entries_fast(data, cfg)

        row = {**params, **per_sym, "min": min(per_sym.values()), "avg": sum(per_sym.values()) / len(per_sym)}
        scored.append(row)
        if row["min"] >= TARGET_ENTRIES:
            hits.append(row)
            print(f"  HIT #{len(hits)}: {params} -> {per_sym}", flush=True)

        if (i + 1) % 36 == 0:
            print(f"  … {i + 1}/{len(combos)} ({time.perf_counter() - t1:.1f}s)", flush=True)

    scored.sort(key=lambda x: (x["min"], x["avg"]), reverse=True)
    hits.sort(key=lambda x: x["min"], reverse=True)

    print(f"\nSearch done in {time.perf_counter() - t1:.1f}s\n")
    print("=== ALL symbols >= 40 entries ===")
    if hits:
        best = hits[0]
        print("RECOMMENDED BB settings (sidebar):")
        for k in keys:
            print(f"  {k}: {best[k]}")
        print("  per_symbol:", {s: best[s] for s in SYMBOLS})
    else:
        print("(none hit 40 on all coins — top 10 by min entries)\n")
        for row in scored[:10]:
            print(row)

    if scored:
        top = scored[0]
        print("\n=== TOP candidate detail ===")
        for k in keys:
            print(f"  {k}: {top[k]}")
        print("  per_symbol:", {s: top[s] for s in SYMBOLS})


if __name__ == "__main__":
    main()
