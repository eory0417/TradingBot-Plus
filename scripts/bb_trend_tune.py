"""추세필터(bb_trend_mode) + Trend Len/% 그리드 탐색 (고속).

다른 설정은 현재 추천값으로 고정. 90일 · 4코인 합산 순손익 기준.
지표·MTF는 심볼당 1회 사전계산 후 재사용.
"""

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

from backtest.engine import (  # noqa: E402
    BacktestCosts,
    _precompute_mtf_bias_series,
    fetch_backtest_frames,
)
from bb_breakout import BB_OHLCV_LIMIT, evaluate_bb_entry  # noqa: E402
from config import settings as app_settings  # noqa: E402
from mtf_filter import gate_entry, parse_mtf_tfs  # noqa: E402
from sizing import SizingState, compute_notional  # noqa: E402
from strategy import Position  # noqa: E402
from trading_engine import compute_indicators_from_df  # noqa: E402

DAYS = 90
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
COSTS = BacktestCosts(fee_pct=0.04, slippage_pct=0.02)
INITIAL = 10_000.0
CACHE_DIR = Path(__file__).resolve().parent / "cache"
FRAMES_KEY = f"frames_mtf_{DAYS}d.pkl"
FAST_KEY = f"trend_tune_fast_{DAYS}d.pkl"
OUT_PATH = Path(__file__).resolve().parent / "bb_trend_tune_out.txt"

FIXED = dict(
    bb_len=20,
    bb_mult=3.2,
    bb_min=1.1,
    vol_mult=1.5,
    vol_len=15,
    min_range_pct=0.05,
    bb_leverage=1,
    bb_max_add_leverage=1,
    position_size_usdt=100.0,
    stop_loss_mode="fixed",
    stop_loss_pct=2.0,
    trailing_profit_pct=1.0,
    trailing_atr_mult=2.0,
    time_exit_hours=3.0,
    mtf_filter_enabled=True,
    mtf_mode="reduce",
    mtf_reduce_mult=0.3,
    scale_out_enabled=True,
    scale_out_fraction=0.4,
    scale_out_atr_mult=1.5,
    scale_out_move_be=True,
    sizing_mode="vol",
    vol_target_atr_pct=0.8,
)

TREND_LENS = [3, 5, 8]
TREND_PCTS = [0.15, 0.25, 0.35]
MODES = ["off", "relaxed", "strict"]


@dataclass
class FastSymbol:
    rows_1m: np.ndarray
    trade_mask: np.ndarray
    ind_at: list
    mtf_series: list


def _combos() -> list[dict]:
    out: list[dict] = []
    for mode in MODES:
        if mode == "off":
            out.append(dict(bb_trend_mode="off", f_trend_len=0, f_trend_pct=0.0))
            continue
        for ln, pct in itertools.product(TREND_LENS, TREND_PCTS):
            out.append(dict(bb_trend_mode=mode, f_trend_len=ln, f_trend_pct=pct))
    return out


def _build_ind_at(rows: np.ndarray, df_15m, symbol: str) -> list:
    ind_ts = df_15m["timestamp"].to_numpy(dtype=np.int64)
    n = len(rows)
    out: list = [None] * n
    last_ind = None
    ind_idx = 0
    for i in range(n):
        ts = int(rows[i, 0])
        while ind_idx + 1 < len(ind_ts) and ind_ts[ind_idx + 1] <= ts:
            ind_idx += 1
            if ind_idx >= 19:
                sub = df_15m.iloc[: ind_idx + 1]
                last_ind = compute_indicators_from_df(symbol, "15m", sub)
        out[i] = last_ind if ind_idx >= 19 else None
    return out


def _load_frames(since: int, until: int) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / FRAMES_KEY
    if path.exists():
        print(f"frames cache hit {path}", flush=True)
        with path.open("rb") as f:
            return pickle.load(f)
    frames: dict = {}
    tfs = parse_mtf_tfs("1h,4h")
    for sym in SYMBOLS:
        print(f"fetch {sym}...", flush=True)
        t0 = time.perf_counter()
        df_1m, df_ind, mtf = fetch_backtest_frames(
            sym, since_ms=since, until_ms=until, indicator_tf="15m",
            testnet=False, mtf_tfs=tfs, mtf_ema_len=200,
        )
        frames[sym] = (df_1m, df_ind, mtf)
        print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    with path.open("wb") as f:
        pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
    return frames


def _build_fast(frames: dict, since: int, until: int) -> dict[str, FastSymbol]:
    path = CACHE_DIR / FAST_KEY
    if path.exists():
        print(f"fast cache hit {path}", flush=True)
        with path.open("rb") as f:
            return pickle.load(f)

    out: dict[str, FastSymbol] = {}
    for sym, (df_1m, df_ind, mtf) in frames.items():
        print(f"precompute {sym}...", flush=True)
        t0 = time.perf_counter()
        rows = df_1m.to_numpy()
        ts = rows[:, 0].astype(np.int64)
        mask = (ts >= since) & (ts <= until)
        ind_at = _build_ind_at(rows, df_ind, sym)
        mtf_series = _precompute_mtf_bias_series(
            mtf, ts, ema_len=int(FIXED.get("mtf_ema_len", 200) or 200),
        )
        out[sym] = FastSymbol(
            rows_1m=rows, trade_mask=mask, ind_at=ind_at, mtf_series=mtf_series,
        )
        print(f"  {sym}: {time.perf_counter()-t0:.1f}s bars={len(rows):,}", flush=True)
    with path.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def _fee(notional: float) -> float:
    return notional * COSTS.fee_pct / 100


def _slip(price: float, side: str, *, entry: bool) -> float:
    if COSTS.slippage_pct <= 0:
        return price
    adj = price * COSTS.slippage_pct / 100
    if entry:
        return price + adj if side == "long" else price - adj
    return price - adj if side == "long" else price + adj


def simulate(data: FastSymbol, cfg) -> tuple[float, int, float]:
    rows = data.rows_1m
    n = len(rows)
    free = INITIAL
    position: Position | None = None
    last_entry_bar: int | None = None
    trades_pnl: list[float] = []
    wins = 0
    sizing_state = SizingState()

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
                position.mark_price = exit_px
                frac = sig.close_fraction
                is_partial = (
                    sig.exit_type == "scale_out"
                    and frac is not None
                    and 0.05 <= float(frac) < 1.0
                    and not position.scale_out_done
                )
                close_notional = position.notional * float(frac) if is_partial else position.notional
                close_margin = position.margin * float(frac) if is_partial else position.margin
                pnl_pct = position.unrealized_pct()
                gross = close_margin * (pnl_pct / 100) * position.leverage
                exit_fee = _fee(close_notional)
                net = gross - exit_fee
                free += close_margin + net
                trades_pnl.append(net)
                if net > 0:
                    wins += 1
                sizing_state.stats.record(pnl_pct)
                if is_partial:
                    position.apply_scale_out(float(frac))
                else:
                    position = None

        window = rows[i - BB_OHLCV_LIMIT + 1 : i + 1]
        entry = evaluate_bb_entry(window, cfg)
        if entry.ok and entry.side and position is None and ind is not None:
            if last_entry_bar == ts:
                continue
            mtf_mult = 1.0
            mtf_ok = True
            if cfg.mtf_filter_enabled and data.mtf_series:
                mtf_ok, mtf_mult, _ = gate_entry(entry.side, data.mtf_series[i], cfg)
            if not mtf_ok:
                continue
            lev = max(1, int(cfg.bb_leverage))
            atr = ind.atr if ind.atr > 0 else close * 0.01
            notional = compute_notional(
                float(cfg.position_size_usdt),
                atr=atr,
                price=close,
                cfg=cfg,
                sizing_state=sizing_state,
                extra_mult=mtf_mult,
            )
            margin = notional / lev
            if free < margin + _fee(notional):
                continue
            entry_px = _slip(close, entry.side, entry=True)
            entry_fee = _fee(notional)
            free -= margin + entry_fee
            position = Position(
                symbol="",
                side=entry.side,
                amount=notional / entry_px,
                entry_price=entry_px,
                atr=atr,
                opened_at=datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                notional=notional,
                leverage=lev,
                margin=margin,
                stop_loss_mode=cfg.stop_loss_mode,
                stop_loss_pct=cfg.stop_loss_pct,
                stop_loss_atr_mult=cfg.stop_loss_atr_mult,
                atr_mult_base=cfg.trailing_atr_mult,
                atr_mult_tight=cfg.trailing_atr_mult_tight,
                trailing_profit_pct=cfg.trailing_profit_pct,
                time_exit_hours=cfg.time_exit_hours,
                scale_out_enabled=cfg.scale_out_enabled,
                scale_out_fraction=cfg.scale_out_fraction,
                scale_out_atr_mult=cfg.scale_out_atr_mult,
                scale_out_move_be=cfg.scale_out_move_be,
            )
            last_entry_bar = ts

    wr = wins / len(trades_pnl) * 100 if trades_pnl else 0.0
    # 미청산은 손익 미포함 (기존 튜닝 스크립트와 동일)
    final = free
    if position is not None:
        final += position.margin  # 증거금만 반환, 미실현 제외
    return final - INITIAL, len(trades_pnl), wr


def _run_combo(cache: dict[str, FastSymbol], overrides: dict) -> dict:
    cfg = app_settings.model_copy(deep=True, update={**FIXED, **overrides})
    total_pnl = 0.0
    total_trades = 0
    wins_w = 0.0
    per: dict[str, dict] = {}
    for sym, data in cache.items():
        pnl, trades, wr = simulate(data, cfg)
        total_pnl += pnl
        total_trades += trades
        wins_w += wr / 100.0 * trades
        per[sym] = {"pnl": pnl, "trades": trades, "wr": wr}
    agg_wr = wins_w / total_trades * 100 if total_trades else 0.0
    return {**overrides, "pnl": total_pnl, "trades": total_trades, "wr": agg_wr, "per": per}


def main() -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS)
    since = int(start.timestamp() * 1000)
    until = int(end.timestamp() * 1000)
    combos = _combos()
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    log(f"BB trend tune (fast) | {DAYS}d | {SYMBOLS}")
    log(
        f"fixed: trail={FIXED['trailing_profit_pct']}% time={FIXED['time_exit_hours']}h "
        f"MTF={FIXED['mtf_mode']} scale={FIXED['scale_out_fraction']} sizing={FIXED['sizing_mode']}"
    )
    log(f"grid: modes={MODES} len={TREND_LENS} pct={TREND_PCTS} → {len(combos)} combos")

    frames = _load_frames(since, until)
    t0 = time.perf_counter()
    cache = _build_fast(frames, since, until)
    log(f"precompute done in {time.perf_counter()-t0:.1f}s")

    t1 = time.perf_counter()
    _ = _run_combo(cache, combos[0])
    one = time.perf_counter() - t1
    log(f"1 combo ≈ {one:.1f}s → 전체 예상 ≈ {one * len(combos) / 60:.1f} min")

    results: list[dict] = []
    t_all = time.perf_counter()
    for i, ov in enumerate(combos, 1):
        t2 = time.perf_counter()
        row = _run_combo(cache, ov)
        results.append(row)
        log(
            f"[{i:02d}/{len(combos)}] {row['bb_trend_mode']:7s} "
            f"len={row['f_trend_len']:2d} pct={row['f_trend_pct']:.2f} | "
            f"PnL {row['pnl']:+7.2f} | trades={row['trades']:3d} WR={row['wr']:5.1f}% "
            f"({time.perf_counter()-t2:.1f}s)"
        )

    results.sort(key=lambda r: r["pnl"], reverse=True)
    log("")
    log("=" * 72)
    log("TOP 10 (합산 순손익)")
    log("=" * 72)
    for i, r in enumerate(results[:10], 1):
        log(
            f"{i:2d}. {r['bb_trend_mode']:7s} len={r['f_trend_len']:2d} "
            f"pct={r['f_trend_pct']:.2f} | "
            f"PnL {r['pnl']:+7.2f} | trades={r['trades']:3d} WR={r['wr']:5.1f}%"
        )
        for sym, p in r["per"].items():
            log(f"      {sym}: {p['pnl']:+.2f} ({p['trades']}t, WR {p['wr']:.1f}%)")

    best = results[0]
    baseline = next(r for r in results if r["bb_trend_mode"] == "off")
    log("")
    log(
        f"BEST: mode={best['bb_trend_mode']} len={best['f_trend_len']} "
        f"pct={best['f_trend_pct']} → {best['pnl']:+.2f} USDT"
    )
    log(
        f"BASELINE (off): {baseline['pnl']:+.2f} USDT | "
        f"delta={best['pnl'] - baseline['pnl']:+.2f}"
    )
    log(f"elapsed {time.perf_counter() - t_all:.1f}s")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
