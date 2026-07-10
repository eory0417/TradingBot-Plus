"""BB 돌파 백테스트 엔진 — 사이드바 설정(bb_breakout + strategy) 기반."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import pandas as pd

from bb_breakout import BB_OHLCV_LIMIT, evaluate_bb_entry
from config import Settings, settings as default_settings
from mtf_filter import gate_entry, parse_mtf_tfs
from sizing import SizingState, compute_notional
from strategy import Position
from trading_engine import compute_indicators_from_df

Side = Literal["long", "short"]

_EXIT_LABELS = {
    "stop_loss": "손절",
    "trailing_stop": "트레일링",
    "time_exit": "시간청산",
    "scale_out": "부분익절",
}

_TF_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

# UI 최대 백테스트 일수 — 1m 약 130k봉 + 워밍업
MAX_BACKTEST_DAYS = 90
_MS_PER_DAY = 86_400_000
_MAX_1M_BARS = 150_000


def _bars_for_span(since_ms: int, until_ms: int, *, bar_ms: int) -> int:
    """구간 길이에 맞는 봉 수(+여유)를 반환한다."""
    span = max(0, until_ms - since_ms)
    return max(1, int(span // bar_ms) + 5)


def _cap_max_bars(needed: int, *, absolute: int = _MAX_1M_BARS) -> int:
    return min(max(needed, 1), absolute)


@dataclass(slots=True)
class BacktestCosts:
    fee_pct: float = 0.04
    slippage_pct: float = 0.02


@dataclass
class Trade:
    symbol: str
    side: Side
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_usdt: float
    fees_usdt: float
    leverage: int
    added: bool
    exit_type: str

    @property
    def exit_label(self) -> str:
        return _EXIT_LABELS.get(self.exit_type, self.exit_type)


@dataclass
class EntryRejectStats:
    """진입 평가 구간(trade_start~end) 내 1m 봉별 필터 탈락 집계."""

    bars_evaluated: int = 0
    no_breakout: int = 0
    bb_width: int = 0
    volume: int = 0
    trend: int = 0
    range_filter: int = 0
    passed: int = 0
    ind_missing: int = 0
    entered: int = 0

    def as_rows(self) -> list[dict]:
        return [
            {"필터": "무돌파", "건수": self.no_breakout, "설명": "종가가 BB 상·하단 밖으로 나가지 않음"},
            {"필터": "BB_WIDTH", "건수": self.bb_width, "설명": "밴드 폭(bb_min) 미달"},
            {"필터": "VOLUME", "건수": self.volume, "설명": f"거래량 배수(vol_mult) 미달"},
            {"필터": "TREND", "건수": self.trend, "설명": "추세 필터 미달"},
            {"필터": "RANGE", "건수": self.range_filter, "설명": "캔들 변동폭(min_range) 미달"},
            {"필터": "✅ 통과", "건수": self.passed, "설명": "모든 BB 진입 필터 통과"},
            {"필터": "15m 지표 없음", "건수": self.ind_missing, "설명": "필터 통과했으나 15m ATR 등 부족으로 미진입"},
            {"필터": "실제 진입", "건수": self.entered, "설명": "신규 포지션 오픈(피라미딩 제외)"},
        ]


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    timestamps: pd.Series = field(default_factory=pd.Series)
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    entry_rejects: EntryRejectStats = field(default_factory=EntryRejectStats)
    final_equity: float = 0.0
    initial_capital: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    expectancy_pct: float = 0.0
    expectancy_usdt: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    gross_profit_usdt: float = 0.0
    gross_loss_usdt: float = 0.0
    bars: int = 0
    symbol: str = ""


def _max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan) * 100
    return float(dd.min()) if not dd.empty else 0.0


def _warmup_start(start: datetime, *, minutes: int) -> datetime:
    return start - timedelta(minutes=minutes)


def _apply_slippage(price: float, side: Side, *, is_entry: bool, slip_pct: float) -> float:
    if slip_pct <= 0:
        return price
    adj = price * slip_pct / 100
    if is_entry:
        return price + adj if side == "long" else price - adj
    return price - adj if side == "long" else price + adj


def _fee_usdt(notional: float, fee_pct: float) -> float:
    return notional * fee_pct / 100 if fee_pct > 0 else 0.0


def _indicators_at(df_15m: pd.DataFrame, symbol: str, ts_ms: int, tf: str):
    sub = df_15m[df_15m["timestamp"] <= ts_ms]
    if len(sub) < 20:
        return None
    return compute_indicators_from_df(symbol, tf, sub)


def _pyramid_enabled(cfg: Settings) -> bool:
    return int(cfg.bb_max_add_leverage) > int(cfg.bb_leverage)


def _precompute_mtf_bias_series(
    mtf_frames: dict[str, pd.DataFrame],
    timestamps_1m: np.ndarray,
    *,
    ema_len: int,
) -> list:
    """1m 각 봉에 대응하는 MTF bias를 포인터 전진으로 미리 계산 (O(n))."""
    from mtf_filter import MtfTrend, bias_from_close_ema, combine_biases

    if not mtf_frames:
        return [MtfTrend(bias="neutral", ready=False, details="")] * len(timestamps_1m)

    # TF별: timestamp, close, rolling ema series
    tf_state: dict[str, dict] = {}
    for tf, df in mtf_frames.items():
        if df is None or df.empty:
            continue
        ts = df["timestamp"].to_numpy(dtype=np.int64)
        closes = df["close"].to_numpy(dtype=float)
        s = pd.Series(closes)
        ema200 = s.ewm(span=ema_len, adjust=False).mean().to_numpy(dtype=float)
        ema50 = s.ewm(span=50, adjust=False).mean().to_numpy(dtype=float)
        tf_state[tf] = {
            "ts": ts,
            "close": closes,
            "ema200": ema200,
            "ema50": ema50,
            "idx": -1,
        }

    out: list = []
    for t in timestamps_1m:
        biases = []
        parts = []
        ready_count = 0
        for tf, st in tf_state.items():
            idx = st["idx"]
            ts_arr = st["ts"]
            while idx + 1 < len(ts_arr) and ts_arr[idx + 1] <= t:
                idx += 1
            st["idx"] = idx
            if idx < 0:
                continue
            close = float(st["close"][idx])
            use_ema = float(st["ema200"][idx])
            ready = idx + 1 >= ema_len
            if not ready:
                use_ema = float(st["ema50"][idx])
            else:
                ready_count += 1
            b = bias_from_close_ema(close, use_ema)
            biases.append(b)
            parts.append(f"{tf}:{b}")
        if not biases:
            out.append(MtfTrend(bias="neutral", ready=False, details=""))
        else:
            out.append(
                MtfTrend(
                    bias=combine_biases(biases),
                    ready=ready_count == len(tf_state),
                    details="; ".join(parts),
                )
            )
    return out


def run_backtest(
    df_1m: pd.DataFrame,
    df_15m: pd.DataFrame,
    symbol: str,
    *,
    cfg: Settings | None = None,
    initial_capital: float = 10_000.0,
    costs: BacktestCosts | None = None,
    trade_start_ms: int = 0,
    trade_end_ms: int = 0,
    mtf_frames: dict[str, pd.DataFrame] | None = None,
) -> BacktestResult:
    """1m BB 진입 + 15m 지표 청산 시뮬레이션 (뉴스 진입 제외)."""
    cfg = cfg or default_settings
    costs = costs or BacktestCosts()
    tf = cfg.timeframe
    sizing_state = SizingState()

    if df_1m.empty:
        return BacktestResult(initial_capital=initial_capital, symbol=symbol)

    frame = df_1m.copy().reset_index(drop=True)
    if trade_end_ms <= 0:
        trade_end_ms = int(frame["timestamp"].iloc[-1])
    if trade_start_ms <= 0:
        trade_start_ms = int(frame["timestamp"].iloc[0])

    free = float(initial_capital)
    position: Position | None = None
    trades: list[Trade] = []
    equity_hist: list[float] = []
    ts_hist: list[int] = []
    rows_1m = frame.to_numpy()
    last_entry_bar: int | None = None
    rejects = EntryRejectStats()
    mtf_frames = mtf_frames or {}
    mtf_series = (
        _precompute_mtf_bias_series(
            mtf_frames, frame["timestamp"].to_numpy(dtype=np.int64),
            ema_len=cfg.mtf_ema_len,
        )
        if cfg.mtf_filter_enabled and mtf_frames
        else None
    )

    for i in range(len(frame)):
        row = rows_1m[i]
        ts = int(row[0])
        close = float(row[4])
        in_range = trade_start_ms <= ts <= trade_end_ms

        ind = _indicators_at(df_15m, symbol, ts, tf) if in_range else None

        if position is not None and in_range:
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
                exit_px = _apply_slippage(
                    close, position.side, is_entry=False, slip_pct=costs.slippage_pct,
                )
                position.mark_price = exit_px
                frac = sig.close_fraction
                is_partial = (
                    sig.exit_type == "scale_out"
                    and frac is not None
                    and 0.05 <= float(frac) < 1.0
                    and not position.scale_out_done
                )
                close_notional = position.notional * float(frac) if is_partial else position.notional
                close_margin = (
                    position.margin * float(frac) if is_partial else position.margin
                )
                pnl_pct = position.unrealized_pct()
                gross_pnl = close_margin * (pnl_pct / 100) * position.leverage
                exit_fee = _fee_usdt(close_notional, costs.fee_pct)
                net_pnl = gross_pnl - exit_fee
                free += close_margin + net_pnl
                trades.append(
                    Trade(
                        symbol=symbol,
                        side=position.side,
                        entry_time=int(position.opened_at.timestamp() * 1000),
                        exit_time=ts,
                        entry_price=position.entry_price,
                        exit_price=exit_px,
                        pnl_pct=pnl_pct,
                        pnl_usdt=net_pnl,
                        fees_usdt=exit_fee,
                        leverage=position.leverage,
                        added=position.added,
                        exit_type=sig.exit_type,
                    )
                )
                sizing_state.stats.record(pnl_pct)
                if is_partial:
                    position.apply_scale_out(float(frac))
                else:
                    position = None

        if in_range and i >= BB_OHLCV_LIMIT - 1:
            rejects.bars_evaluated += 1
            window = rows_1m[i - BB_OHLCV_LIMIT + 1 : i + 1]
            result = evaluate_bb_entry(window, cfg)
            if not result.side:
                rejects.no_breakout += 1
            elif not result.ok:
                if result.reason == "BB_WIDTH":
                    rejects.bb_width += 1
                elif result.reason == "VOLUME":
                    rejects.volume += 1
                elif result.reason == "TREND":
                    rejects.trend += 1
                elif result.reason == "RANGE":
                    rejects.range_filter += 1
            elif result.ok and result.side:
                rejects.passed += 1
                if position is None:
                    if ind is None:
                        rejects.ind_missing += 1
                    elif last_entry_bar != ts:
                        mtf_mult = 1.0
                        mtf_ok = True
                        if mtf_series is not None:
                            trend = mtf_series[i]
                            mtf_ok, mtf_mult, _reason = gate_entry(result.side, trend, cfg)
                        if mtf_ok:
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
                            if free >= margin:
                                rejects.entered += 1
                                entry_px = _apply_slippage(
                                    close, result.side, is_entry=True, slip_pct=costs.slippage_pct,
                                )
                                entry_fee = _fee_usdt(notional, costs.fee_pct)
                                free -= margin + entry_fee
                                position = Position(
                                    symbol=symbol,
                                    side=result.side,
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
                elif (
                    _pyramid_enabled(cfg)
                    and not position.added
                    and result.side == position.side
                    and ind is not None
                ):
                    favorable = (
                        (position.side == "long" and ind.last_price > position.entry_price)
                        or (position.side == "short" and ind.last_price < position.entry_price)
                    )
                    if favorable and last_entry_bar != ts:
                        mtf_mult = 1.0
                        mtf_ok = True
                        if mtf_series is not None:
                            trend = mtf_series[i]
                            mtf_ok, mtf_mult, _reason = gate_entry(result.side, trend, cfg)
                        if mtf_ok:
                            add_lev = min(position.leverage + 1, int(cfg.bb_max_add_leverage))
                            atr = ind.atr if ind.atr > 0 else close * 0.01
                            add_notional = compute_notional(
                                float(cfg.position_size_usdt),
                                atr=atr,
                                price=close,
                                cfg=cfg,
                                sizing_state=sizing_state,
                                extra_mult=mtf_mult,
                            )
                            add_margin = add_notional / add_lev
                            if free >= add_margin:
                                add_px = _apply_slippage(
                                    close, position.side, is_entry=True, slip_pct=costs.slippage_pct,
                                )
                                add_fee = _fee_usdt(add_notional, costs.fee_pct)
                                free -= add_margin + add_fee
                                add_amount = add_notional / add_px
                                position.add_fill(
                                    add_amount=add_amount,
                                    add_price=add_px,
                                    add_notional=add_notional,
                                    add_margin=add_margin,
                                    leverage=add_lev,
                                    atr=atr,
                                )
                                last_entry_bar = ts

        mark_eq = free
        if position is not None:
            position.mark_price = close
            unreal = position.margin * (position.unrealized_pct() / 100) * position.leverage
            mark_eq = free + position.margin + unreal
        equity_hist.append(mark_eq)
        ts_hist.append(ts)

    wins = [t for t in trades if t.pnl_usdt > 0]
    losses = [t for t in trades if t.pnl_usdt <= 0]
    gross_profit = sum(t.pnl_usdt for t in wins)
    gross_loss = abs(sum(t.pnl_usdt for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    final = equity_hist[-1] if equity_hist else free
    eq_series = pd.Series(equity_hist, index=frame.index)
    expectancy_usdt = sum(t.pnl_usdt for t in trades) / len(trades) if trades else 0.0
    expectancy_pct = sum(t.pnl_pct for t in trades) / len(trades) if trades else 0.0

    return BacktestResult(
        trades=trades,
        equity_curve=eq_series,
        timestamps=pd.Series(ts_hist, index=frame.index),
        frame=frame,
        final_equity=final,
        initial_capital=initial_capital,
        win_rate=len(wins) / len(trades) * 100 if trades else 0.0,
        total_trades=len(trades),
        profit_factor=pf,
        expectancy_pct=expectancy_pct,
        expectancy_usdt=expectancy_usdt,
        total_return_pct=(final / initial_capital - 1) * 100 if initial_capital else 0.0,
        max_drawdown_pct=_max_drawdown_pct(eq_series),
        avg_win_pct=sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0,
        avg_loss_pct=-sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0,
        gross_profit_usdt=gross_profit,
        gross_loss_usdt=gross_loss,
        bars=len(frame),
        symbol=symbol,
        entry_rejects=rejects,
    )


def _make_exchange(*, testnet: bool):
    import ccxt

    ex = ccxt.binance(
        {"enableRateLimit": True, "options": {"defaultType": "future"}},
    )
    if testnet:
        try:
            ex.enable_demo_trading(True)
        except Exception:
            pass
    ex.load_markets()
    return ex


def _unify_symbol(symbol: str) -> str:
    if ":" in symbol:
        return symbol
    if "/" in symbol:
        return f"{symbol}:USDT" if not symbol.endswith(":USDT") else symbol
    return f"{symbol}/USDT:USDT"


def fetch_ohlcv_sync(
    symbol: str,
    timeframe: str,
    limit: int = 1500,
    *,
    testnet: bool = False,
) -> pd.DataFrame:
    """ccxt 동기로 OHLCV 조회 (최근 N봉)."""
    ex = _make_exchange(testnet=testnet)
    unified = _unify_symbol(symbol)
    rows = ex.fetch_ohlcv(unified, timeframe=timeframe, limit=limit)
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def fetch_ohlcv_range(
    symbol: str,
    timeframe: str,
    *,
    since_ms: int,
    until_ms: int | None = None,
    max_bars: int | None = None,
    testnet: bool = False,
) -> pd.DataFrame:
    """기간 지정 OHLCV (페이지네이션). ``max_bars`` 미지정 시 구간 전체를 가져온다."""
    ex = _make_exchange(testnet=testnet)
    unified = _unify_symbol(symbol)
    until_ms = until_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    bar_ms = _TF_MS.get(timeframe, 60_000)
    if max_bars is None:
        max_bars = _bars_for_span(since_ms, until_ms, bar_ms=bar_ms)

    all_rows: list[list] = []
    since = since_ms
    page_limit = 1000  # Binance USD-M futures OHLCV 최대 1000봉/요청
    while len(all_rows) < max_bars:
        batch = ex.fetch_ohlcv(unified, timeframe=timeframe, since=since, limit=page_limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= until_ms:
            break
        since = last_ts + 1

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df[(df["timestamp"] >= since_ms) & (df["timestamp"] <= until_ms)]
    if len(df) > max_bars:
        df = df.iloc[-max_bars:].reset_index(drop=True)
    return df


def fetch_backtest_frames(
    symbol: str,
    *,
    since_ms: int,
    until_ms: int,
    indicator_tf: str,
    testnet: bool = False,
    mtf_tfs: list[str] | None = None,
    mtf_ema_len: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """1m(진입) + 지표 TF + (선택) MTF OHLCV를 워밍업 포함해 조회한다."""
    warmup_1m = BB_OHLCV_LIMIT + 20
    tf_ms = _TF_MS.get(indicator_tf, 900_000)
    warmup_15m = max(60, int(20 * tf_ms / 60_000))
    since_1m = since_ms - warmup_1m * 60_000
    since_ind = since_ms - warmup_15m * tf_ms

    need_1m = _cap_max_bars(_bars_for_span(since_1m, until_ms, bar_ms=60_000))
    need_ind = _cap_max_bars(_bars_for_span(since_ind, until_ms, bar_ms=tf_ms), absolute=20_000)

    df_1m = fetch_ohlcv_range(
        symbol, "1m", since_ms=since_1m, until_ms=until_ms,
        max_bars=need_1m, testnet=testnet,
    )
    df_ind = fetch_ohlcv_range(
        symbol, indicator_tf, since_ms=since_ind, until_ms=until_ms,
        max_bars=need_ind, testnet=testnet,
    )
    mtf: dict[str, pd.DataFrame] = {}
    for tf in (mtf_tfs or []):
        bar_ms = _TF_MS.get(tf, 3_600_000)
        since_mtf = since_ms - max(mtf_ema_len + 20, 80) * bar_ms
        need = _cap_max_bars(_bars_for_span(since_mtf, until_ms, bar_ms=bar_ms), absolute=20_000)
        mtf[tf] = fetch_ohlcv_range(
            symbol, tf, since_ms=since_mtf, until_ms=until_ms,
            max_bars=need, testnet=testnet,
        )
    return df_1m, df_ind, mtf
