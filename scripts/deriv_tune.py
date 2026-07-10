"""펀딩/OI 필터 그리드 탐색 — 90일 · 4코인 BB 백테스트 합산 PnL.

나머지 설정은 현재 추천값 고정. MTF 직후 deriv 게이트를 시뮬레이션에 반영.

주의: 현재 derivatives_filter.py 에서 OI_SPIKE_PCT 는 로그 보조용이며
차단/축소 판정에는 영향 없음 → 이 스크립트는 **펀딩 임계·모드·축소배수** 위주.
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
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from backtest.engine import _make_exchange, _unify_symbol, fetch_backtest_frames  # noqa: E402
from bb_breakout import BB_OHLCV_LIMIT, evaluate_bb_entry  # noqa: E402
from bb_trend_tune import (  # noqa: E402
    CACHE_DIR,
    DAYS,
    FRAMES_KEY,
    SYMBOLS,
    _build_ind_at,
    _fee,
    _slip,
    COSTS,
    INITIAL,
)
from config import Settings, settings as app_settings  # noqa: E402
from derivatives_filter import DerivativesSnapshot, gate_entry  # noqa: E402
from mtf_filter import gate_entry as mtf_gate_entry, parse_mtf_tfs  # noqa: E402
from backtest.engine import _precompute_mtf_bias_series  # noqa: E402
from sizing import SizingState, compute_notional  # noqa: E402
from strategy import Position  # noqa: E402
from trading_engine import compute_indicators_from_df  # noqa: E402

OUT_PATH = SCRIPTS / "deriv_tune_out.txt"
DERIV_CACHE = CACHE_DIR / f"deriv_funding_{DAYS}d.pkl"
FAST_KEY = f"deriv_tune_fast_{DAYS}d.pkl"

# 현재 추천값 고정 (BB LMM + trend tune 결과)
FIXED = dict(
    bb_len=18,
    bb_mult=3.7,
    bb_min=0.7,
    vol_mult=1.5,
    vol_len=15,
    min_range_pct=0.05,
    bb_trend_mode="relaxed",
    f_trend_len=5,
    f_trend_pct=0.15,
    bb_leverage=1,
    bb_max_add_leverage=1,
    position_size_usdt=100.0,
    stop_loss_mode="fixed",
    stop_loss_pct=2.0,
    trailing_profit_pct=1.0,
    trailing_atr_mult=2.0,
    trailing_atr_mult_tight=1.5,
    time_exit_hours=3.0,
    mtf_filter_enabled=True,
    mtf_mode="reduce",
    mtf_reduce_mult=0.3,
    mtf_tfs="1h,4h",
    mtf_ema_len=200,
    scale_out_enabled=True,
    scale_out_fraction=0.4,
    scale_out_atr_mult=1.5,
    scale_out_move_be=True,
    sizing_mode="vol",
    vol_target_atr_pct=0.8,
    sizing_min_mult=0.4,
    sizing_max_mult=1.5,
    deriv_require_ready=False,
    oi_spike_pct=5.0,
    oi_cache_sec=300,
)

FUNDING_THRESH_PCT = [0.02, 0.03, 0.05, 0.08, 0.10]
REDUCE_MULTS = [0.33, 0.5, 0.7]


@dataclass
class DerivFastSymbol:
    rows_1m: np.ndarray
    trade_mask: np.ndarray
    ind_at: list
    mtf_series: list
    funding_pct_at: np.ndarray  # per 1m bar, funding rate in %


def _fetch_funding_history(symbol: str, since_ms: int, until_ms: int) -> tuple[np.ndarray, np.ndarray]:
    ex = _make_exchange(testnet=False)
    unified = _unify_symbol(symbol)
    all_rows: list[dict] = []
    since = since_ms - 7 * 86_400_000  # 워밍업
    while True:
        batch = ex.fetch_funding_rate_history(unified, since=since, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = int(batch[-1]["timestamp"])
        if last_ts >= until_ms or len(batch) < 1000:
            break
        since = last_ts + 1
        time.sleep(ex.rateLimit / 1000 if ex.rateLimit else 0.2)

    if not all_rows:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    by_ts: dict[int, float] = {}
    for row in all_rows:
        ts = int(row["timestamp"])
        if since_ms <= ts <= until_ms or ts < since_ms:
            by_ts[ts] = float(row["fundingRate"]) * 100.0
    if not by_ts:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    times = np.array(sorted(by_ts.keys()), dtype=np.int64)
    rates = np.array([by_ts[t] for t in times], dtype=np.float64)
    return times, rates


def _align_funding(bar_ts: np.ndarray, fr_times: np.ndarray, fr_rates: np.ndarray) -> np.ndarray:
    n = len(bar_ts)
    out = np.zeros(n, dtype=np.float64)
    if fr_times.size == 0:
        return out
    idx = np.searchsorted(fr_times, bar_ts, side="right") - 1
    idx = np.clip(idx, 0, len(fr_rates) - 1)
    out = fr_rates[idx]
    out[bar_ts < fr_times[0]] = fr_rates[0]
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
        frames[sym] = fetch_backtest_frames(
            sym, since_ms=since, until_ms=until, indicator_tf="15m",
            testnet=False, mtf_tfs=tfs, mtf_ema_len=200,
        )
        print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    with path.open("wb") as f:
        pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
    return frames


def _load_funding_cache(since: int, until: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if DERIV_CACHE.exists():
        print(f"funding cache hit {DERIV_CACHE}", flush=True)
        with DERIV_CACHE.open("rb") as f:
            return pickle.load(f)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym in SYMBOLS:
        print(f"funding fetch {sym}...", flush=True)
        t0 = time.perf_counter()
        out[sym] = _fetch_funding_history(sym, since, until)
        print(
            f"  {sym}: {len(out[sym][0])} points in {time.perf_counter()-t0:.1f}s",
            flush=True,
        )
    with DERIV_CACHE.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def _build_fast(frames: dict, since: int, until: int, funding: dict) -> dict[str, DerivFastSymbol]:
    path = CACHE_DIR / FAST_KEY
    if path.exists():
        print(f"deriv fast cache hit {path}", flush=True)
        with path.open("rb") as f:
            return pickle.load(f)

    out: dict[str, DerivFastSymbol] = {}
    for sym, (df_1m, df_ind, mtf) in frames.items():
        print(f"precompute {sym}...", flush=True)
        t0 = time.perf_counter()
        rows = df_1m.to_numpy()
        ts = rows[:, 0].astype(np.int64)
        mask = (ts >= since) & (ts <= until)
        ind_at = _build_ind_at(rows, df_ind, sym)
        mtf_series = _precompute_mtf_bias_series(mtf, ts, ema_len=200)
        fr_times, fr_rates = funding.get(sym, (np.array([]), np.array([])))
        funding_at = _align_funding(ts, fr_times, fr_rates)
        out[sym] = DerivFastSymbol(
            rows_1m=rows,
            trade_mask=mask,
            ind_at=ind_at,
            mtf_series=mtf_series,
            funding_pct_at=funding_at,
        )
        print(f"  {sym}: {time.perf_counter()-t0:.1f}s", flush=True)
    with path.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def _combos() -> list[dict]:
    out: list[dict] = [dict(deriv_filter_enabled=False, deriv_mode="reduce", deriv_reduce_mult=0.5,
                            funding_long_block_pct=0.05, funding_short_block_pct=-0.05, label="off")]
    for th in FUNDING_THRESH_PCT:
        out.append(dict(
            deriv_filter_enabled=True,
            deriv_mode="block",
            deriv_reduce_mult=0.5,
            funding_long_block_pct=th,
            funding_short_block_pct=-th,
            label=f"block@{th}",
        ))
    for th, mult in itertools.product(FUNDING_THRESH_PCT, REDUCE_MULTS):
        out.append(dict(
            deriv_filter_enabled=True,
            deriv_mode="reduce",
            deriv_reduce_mult=mult,
            funding_long_block_pct=th,
            funding_short_block_pct=-th,
            label=f"reduce@{th}x{mult}",
        ))
    return out


def simulate(data: DerivFastSymbol, cfg: Settings) -> tuple[float, int, float, int]:
    rows = data.rows_1m
    n = len(rows)
    free = INITIAL
    position: Position | None = None
    last_entry_bar: int | None = None
    trades_pnl: list[float] = []
    wins = 0
    deriv_blocks = 0
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
            extra_mult = 1.0
            if cfg.mtf_filter_enabled and data.mtf_series:
                mtf_ok, mtf_mult, _ = mtf_gate_entry(entry.side, data.mtf_series[i], cfg)
                if not mtf_ok:
                    continue
                extra_mult *= mtf_mult

            if cfg.deriv_filter_enabled:
                snap = DerivativesSnapshot(
                    funding_rate_pct=float(data.funding_pct_at[i]),
                    open_interest=1.0,
                    oi_change_pct=0.0,
                    ready=True,
                )
                d_ok, d_mult, _ = gate_entry(entry.side, snap, cfg)
                if not d_ok:
                    deriv_blocks += 1
                    continue
                extra_mult *= d_mult

            lev = max(1, int(cfg.bb_leverage))
            atr = ind.atr if ind.atr > 0 else close * 0.01
            notional = compute_notional(
                float(cfg.position_size_usdt),
                atr=atr,
                price=close,
                cfg=cfg,
                sizing_state=sizing_state,
                extra_mult=extra_mult,
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
    final = free
    if position is not None:
        final += position.margin
    return final - INITIAL, len(trades_pnl), wr, deriv_blocks


def _run_combo(cache: dict[str, DerivFastSymbol], overrides: dict) -> dict:
    label = overrides.pop("label", "")
    cfg = app_settings.model_copy(deep=True, update={**FIXED, **overrides})
    total_pnl = 0.0
    total_trades = 0
    total_blocks = 0
    wins_w = 0.0
    per: dict[str, dict] = {}
    for sym, data in cache.items():
        pnl, trades, wr, blocks = simulate(data, cfg)
        total_pnl += pnl
        total_trades += trades
        total_blocks += blocks
        wins_w += wr / 100.0 * trades
        per[sym] = {"pnl": pnl, "trades": trades, "wr": wr, "blocks": blocks}
    agg_wr = wins_w / total_trades * 100 if total_trades else 0.0
    return {
        "label": label,
        **overrides,
        "pnl": total_pnl,
        "trades": total_trades,
        "wr": agg_wr,
        "deriv_blocks": total_blocks,
        "per": per,
    }


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

    log(f"Deriv funding tune | {DAYS}d | {SYMBOLS}")
    log(
        f"fixed: BB {FIXED['bb_len']}/{FIXED['bb_mult']}/{FIXED['bb_min']} "
        f"trend={FIXED['bb_trend_mode']} len={FIXED['f_trend_len']} "
        f"MTF={FIXED['mtf_mode']} scale={FIXED['scale_out_fraction']}"
    )
    log(
        f"grid: thresh={FUNDING_THRESH_PCT} reduce_mult={REDUCE_MULTS} "
        f"→ {len(combos)} combos (OI_SPIKE 미반영 — 현재 코드 한계)"
    )

    frames = _load_frames(since, until)
    funding = _load_funding_cache(since, until)
    cache = _build_fast(frames, since, until, funding)

    results: list[dict] = []
    t_all = time.perf_counter()
    for i, raw in enumerate(combos, 1):
        overrides = dict(raw)
        t0 = time.perf_counter()
        r = _run_combo(cache, overrides)
        results.append(r)
        log(
            f"[{i}/{len(combos)}] {r['label']:18s} "
            f"pnl={r['pnl']:+8.2f} trades={r['trades']:4d} wr={r['wr']:5.1f}% "
            f"blocks={r['deriv_blocks']:4d} ({time.perf_counter()-t0:.1f}s)"
        )

    results.sort(key=lambda x: x["pnl"], reverse=True)
    baseline = next((x for x in results if x.get("label") == "off"), results[-1])
    best = results[0]

    log("")
    log(f"Total time: {time.perf_counter()-t_all:.1f}s")
    log(f"Baseline (deriv off): pnl={baseline['pnl']:+.2f} trades={baseline['trades']}")
    log(f"Best: {best['label']} pnl={best['pnl']:+.2f} Δ={best['pnl']-baseline['pnl']:+.2f}")
    log(
        f"  → mode={best.get('deriv_mode')} thresh=±{best.get('funding_long_block_pct')} "
        f"reduce_mult={best.get('deriv_reduce_mult')}"
    )
    log("")
    log("Top 10:")
    for r in results[:10]:
        log(
            f"  {r['label']:18s} pnl={r['pnl']:+8.2f} trades={r['trades']:4d} "
            f"wr={r['wr']:5.1f}% blocks={r['deriv_blocks']:4d}"
        )

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
