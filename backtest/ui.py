"""Streamlit 백테스트 UI (TradingView Strategy Tester 스타일)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bb_breakout import BB_OHLCV_LIMIT, bb_series_for_chart
from backtest.engine import (
    BacktestCosts,
    BacktestResult,
    MAX_BACKTEST_DAYS,
    Trade,
    fetch_backtest_frames,
    run_backtest,
)
from config import Settings, settings
from kst_util import TZ_LABEL, format_kst, series_ms_to_kst_pandas

_TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")
_PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#161B22",
    font=dict(color="#E6EDF3", size=10),
)

_EXIT_LABELS = {
    "stop_loss": "손절",
    "trailing_stop": "트레일링",
    "time_exit": "시간청산",
}


def _symbol_options() -> list[str]:
    return list(settings.symbols) or ["BTC/USDT"]


def _settings_caption(cfg: Settings) -> str:
    sl = (
        f"손절 {cfg.stop_loss_pct}%"
        if cfg.stop_loss_mode == "fixed"
        else f"손절 ATR×{cfg.stop_loss_atr_mult:g}"
    )
    trend = {"relaxed": "완화", "strict": "엄격", "off": "끔"}.get(cfg.bb_trend_mode, cfg.bb_trend_mode)
    pyramid = "ON" if cfg.bb_max_add_leverage > cfg.bb_leverage else "OFF"
    return (
        f"BB({cfg.bb_len},{cfg.bb_mult:g}) · bb_min={cfg.bb_min}% · vol={cfg.vol_mult} · "
        f"{sl} · trail={cfg.trailing_profit_pct}%/ATR×{cfg.trailing_atr_mult:g} · "
        f"time={cfg.time_exit_hours:g}h · trend={trend} · 피라미딩={pyramid} · "
        f"진입금={cfg.position_size_usdt} · BB레버={cfg.bb_leverage}x"
    )


def _build_equity_chart(result: BacktestResult) -> go.Figure:
    times = series_ms_to_kst_pandas(result.timestamps)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=result.equity_curve,
            mode="lines",
            name="자산",
            line=dict(color="#58a6ff", width=2),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.12)",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=40, r=20, t=30, b=30),
        title=dict(text="자산 곡선 (Equity)", font=dict(size=12)),
        yaxis_title="USDT",
        xaxis_title=f"시간 ({TZ_LABEL})",
        showlegend=False,
        **_PLOTLY_DARK,
    )
    return fig


def _build_price_chart(result: BacktestResult, cfg: Settings) -> go.Figure | None:
    frame = result.frame
    if frame is None or frame.empty:
        return None

    closes = frame["close"].tolist()
    upper, basis, lower = bb_series_for_chart(closes, cfg.bb_len, cfg.bb_mult)
    times = series_ms_to_kst_pandas(frame["timestamp"])

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )
    fig.add_trace(
        go.Candlestick(
            x=times,
            open=frame["open"], high=frame["high"],
            low=frame["low"], close=frame["close"],
            name="OHLC",
            increasing_line_color="#3fb950",
            decreasing_line_color="#f85149",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=upper, mode="lines", name="BB Upper",
                   line=dict(width=1, color="#a371f7"), opacity=0.8),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=basis, mode="lines", name="BB Basis",
                   line=dict(width=1, dash="dot", color="#8b949e"), opacity=0.7),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=lower, mode="lines", name="BB Lower",
                   line=dict(width=1, color="#a371f7"), opacity=0.8),
        row=1, col=1,
    )

    def _markers(trades: list[Trade], side: str) -> None:
        if not trades:
            return
        color = "#3fb950" if side == "long" else "#f85149"
        sym = "triangle-up" if side == "long" else "triangle-down"
        xs = series_ms_to_kst_pandas(pd.Series([t.entry_time for t in trades]))
        fig.add_trace(
            go.Scatter(
                x=xs, y=[t.entry_price for t in trades], mode="markers",
                name=f"{side} 진입",
                marker=dict(symbol=sym, size=10, color=color, line=dict(width=1, color="#fff")),
            ),
            row=1, col=1,
        )
        xs2 = series_ms_to_kst_pandas(pd.Series([t.exit_time for t in trades]))
        fig.add_trace(
            go.Scatter(
                x=xs2, y=[t.exit_price for t in trades], mode="markers",
                name=f"{side} 청산",
                marker=dict(symbol="x", size=8, color=color),
            ),
            row=1, col=1,
        )

    long_e = [t for t in result.trades if t.side == "long"]
    short_e = [t for t in result.trades if t.side == "short"]
    _markers(long_e, "long")
    _markers(short_e, "short")

    vol_colors = [
        "#3fb950" if c >= o else "#f85149"
        for c, o in zip(frame["close"], frame["open"])
    ]
    fig.add_trace(
        go.Bar(x=times, y=frame["volume"], marker_color=vol_colors, name="Volume", opacity=0.5),
        row=2, col=1,
    )

    fig.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=40, b=20),
        title=dict(text=f"{result.symbol} · BB 돌파 · 진입/청산 (1m)", font=dict(size=12)),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=9)),
        **_PLOTLY_DARK,
    )
    fig.update_yaxes(title_text="가격", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    return fig


def _trades_dataframe(result: BacktestResult) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(result.trades, 1):
        rows.append({
            "#": i,
            "방향": t.side.upper(),
            "배율": f"{t.leverage}x",
            "추가": "➕" if t.added else "",
            "진입": format_kst(datetime.fromtimestamp(t.entry_time / 1000, tz=timezone.utc)),
            "청산": format_kst(datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc)),
            "진입가": round(t.entry_price, 4),
            "청산가": round(t.exit_price, 4),
            "손익%": round(t.pnl_pct, 2),
            "손익USDT": round(t.pnl_usdt, 2),
            "수수료": round(t.fees_usdt, 2),
            "사유": t.exit_label,
        })
    return pd.DataFrame(rows)


def _entry_reject_stats(result: BacktestResult) -> pd.DataFrame:
    rej = result.entry_rejects
    if rej.bars_evaluated <= 0:
        return pd.DataFrame()
    rows = rej.as_rows()
    df = pd.DataFrame(rows)
    df["비율%"] = (df["건수"] / rej.bars_evaluated * 100).round(2)
    return df


def _exit_stats(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    by_type: dict[str, dict] = {}
    for t in trades:
        key = t.exit_type
        if key not in by_type:
            by_type[key] = {"사유": t.exit_label, "건수": 0, "손익USDT": 0.0}
        by_type[key]["건수"] += 1
        by_type[key]["손익USDT"] += t.pnl_usdt
    rows = list(by_type.values())
    for r in rows:
        r["손익USDT"] = round(r["손익USDT"], 2)
    return pd.DataFrame(rows)


def render_backtest_tab() -> None:
    """백테스트 탭 — 사이드바 BB·청산 설정을 그대로 사용."""
    st.markdown("##### 📉 백테스트 (TradingView Strategy Tester 유사)")
    st.caption(
        "사이드바 **BB 진입·손절·트레일링·시간청산** 설정이 그대로 적용됩니다. "
        "1m BB 돌파 진입 + 15m 지표 청산 시뮬레이션 (뉴스 진입 제외)."
    )
    st.caption(f"적용 설정: {_settings_caption(settings)}")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    symbol = c1.selectbox("심볼", _symbol_options(), key="bt_symbol")
    mode = c2.radio("데이터", ["최근 N일", "기간 지정"], horizontal=True, key="bt_mode")
    use_testnet = c3.checkbox("Demo 시세", value=False, help="체크 시 testnet 시세 (기본: 메인넷 공개)")
    use_costs = c4.checkbox("수수료·슬리피지", value=True, key="bt_use_costs")

    r1, r2, r3, r4 = st.columns(4)
    capital = r1.number_input("초기 자본 (USDT)", 100.0, 1_000_000.0, 10_000.0, key="bt_cap")
    fee_pct = r2.number_input(
        "편도 수수료 %", 0.0, 1.0, 0.04, step=0.01,
        disabled=not use_costs, key="bt_comm",
    )
    slip_pct = r3.number_input(
        "슬리피지 %", 0.0, 1.0, 0.02, step=0.01,
        disabled=not use_costs, key="bt_slip",
    )

    today = datetime.now(timezone.utc).date()
    if mode == "최근 N일":
        bar_days = r4.number_input(
            "기간 (일)", 7, MAX_BACKTEST_DAYS, 30, step=1, key="bt_days",
            help=f"최대 {MAX_BACKTEST_DAYS}일 · 1m 전구간 다운로드",
        )
        start_d = today - timedelta(days=int(bar_days))
        end_d = today
    else:
        d1, d2 = st.columns(2)
        start_d = d1.date_input("시작일", value=today - timedelta(days=30), key="bt_start")
        end_d = d2.date_input("종료일", value=today, key="bt_end")
        span_days = (end_d - start_d).days
        if span_days > MAX_BACKTEST_DAYS:
            st.warning(f"기간이 {span_days}일입니다. 최대 {MAX_BACKTEST_DAYS}일까지만 권장합니다 (1m 약 {MAX_BACKTEST_DAYS * 1440:,}봉).")
        elif span_days < 1:
            st.error("종료일은 시작일 이후여야 합니다.")
            return

    run = st.button("▶ 백테스트 실행", type="primary", key="bt_run")

    if not run:
        st.info(
            f"설정 후 **백테스트 실행**을 누르세요. "
            f"{MAX_BACKTEST_DAYS}일 기준 1m 약 {MAX_BACKTEST_DAYS * 1440:,}봉 — "
            "첫 실행 시 시세 다운로드에 수 분 걸릴 수 있습니다."
        )
        return

    costs = BacktestCosts(
        fee_pct=fee_pct if use_costs else 0.0,
        slippage_pct=slip_pct if use_costs else 0.0,
    )
    cfg = settings.model_copy(deep=True)

    span_days = max(1, (end_d - start_d).days)
    est_1m = span_days * 1440 + 150
    with st.spinner(
        f"{symbol} 시세 로딩 (1m ~{est_1m:,}봉 + {cfg.timeframe})… "
        f"{'90일 전체는 1~3분 소요될 수 있습니다' if span_days >= 60 else ''}"
    ):
        try:
            if mode == "최근 N일":
                until = int(datetime.combine(end_d, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
                since = int(datetime.combine(start_d, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
                df_1m, df_ind = fetch_backtest_frames(
                    symbol, since_ms=since, until_ms=until,
                    indicator_tf=cfg.timeframe, testnet=use_testnet,
                )
            else:
                since = int(datetime.combine(start_d, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
                until = int(
                    (datetime.combine(end_d, datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)).timestamp() * 1000
                )
                df_1m, df_ind = fetch_backtest_frames(
                    symbol, since_ms=since, until_ms=until,
                    indicator_tf=cfg.timeframe, testnet=use_testnet,
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"시세 조회 실패: {exc}")
            return

    if df_1m.empty or len(df_1m) < BB_OHLCV_LIMIT:
        st.warning(f"1m 데이터 부족 (수신 {len(df_1m)}봉, 필요 최소 {BB_OHLCV_LIMIT}봉). 기간을 늘려 주세요.")
        return

    warmup_bars = BB_OHLCV_LIMIT + 20
    expected_trade_bars = span_days * 1440
    expected_total = expected_trade_bars + warmup_bars
    loaded = len(df_1m)
    first_ts = int(df_1m["timestamp"].iloc[0])
    last_ts = int(df_1m["timestamp"].iloc[-1])
    if loaded < expected_trade_bars * 0.5:
        st.error(
            f"**1m 데이터가 {loaded:,}봉만 로드되었습니다** "
            f"(선택 {span_days}일 → 예상 약 {expected_total:,}봉). "
            f"구간: {format_kst(datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc))} "
            f"~ {format_kst(datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc))}. "
            "터미널에서 Streamlit을 **완전히 중지(Ctrl+C) 후 재실행**해 주세요. "
            f"({'Demo 시세' if use_testnet else '메인넷 공개 시세'})"
        )
        return

    trade_start = since if mode == "기간 지정" else int(
        datetime.combine(start_d, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000
    )
    trade_end = until - 1 if mode == "기간 지정" else int(df_1m["timestamp"].iloc[-1])

    with st.spinner("BB 돌파 시뮬레이션…"):
        try:
            result = run_backtest(
                df_1m,
                df_ind,
                symbol,
                cfg=cfg,
                initial_capital=float(capital),
                costs=costs,
                trade_start_ms=trade_start,
                trade_end_ms=trade_end,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"백테스트 오류: {exc}")
            return

    net = result.final_equity - result.initial_capital
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("순손익", f"{net:+,.2f} USDT")
    m2.metric("총 수익률", f"{result.total_return_pct:+.2f}%")
    m3.metric("최대 낙폭", f"{result.max_drawdown_pct:.2f}%")
    m4.metric("승률", f"{result.win_rate:.1f}%", f"{result.total_trades}건")
    pf = result.profit_factor
    m5.metric("Profit Factor", "∞" if pf == float("inf") else f"{pf:.2f}")
    m6.metric("Expectancy", f"{result.expectancy_usdt:+.2f} USDT")
    m7.metric("분석 봉", f"{result.bars:,}")

    st.caption(
        f"로드 1m {loaded:,}봉 · 평가 구간 {result.entry_rejects.bars_evaluated:,}봉 · "
        f"기간 {start_d} ~ {end_d} · 평균 승 {result.avg_win_pct:+.2f}% / "
        f"평균 패 {result.avg_loss_pct:.2f}% · 거래당 기대값 {result.expectancy_pct:+.3f}%"
    )

    eq_col, stat_col = st.columns([3, 1])
    with eq_col:
        st.plotly_chart(_build_equity_chart(result), width="stretch")
    with stat_col:
        st.markdown("**요약**")
        st.write(f"초기: {result.initial_capital:,.2f} USDT")
        st.write(f"최종: {result.final_equity:,.2f} USDT")
        st.write(f"총 이익: {result.gross_profit_usdt:+,.2f} USDT")
        st.write(f"총 손실: -{result.gross_loss_usdt:,.2f} USDT")

    price_fig = _build_price_chart(result, cfg)
    if price_fig is not None:
        st.plotly_chart(price_fig, width="stretch")

    reject_df = _entry_reject_stats(result)
    if not reject_df.empty:
        st.markdown("##### 진입 필터별 탈락 (1m 봉 평가)")
        st.caption(
            f"분석 구간 {result.entry_rejects.bars_evaluated:,}봉 평가 · "
            "돌파 후 필터 탈락은 방향(롱/숏)은 잡힌 뒤 차단된 경우입니다."
        )
        st.dataframe(reject_df, width="stretch", hide_index=True)

    exit_df = _exit_stats(result.trades)
    if not exit_df.empty:
        st.markdown("##### 청산 사유별 손익")
        st.dataframe(exit_df, width="stretch", hide_index=True)

    if result.trades:
        st.markdown("##### 거래 내역")
        st.dataframe(_trades_dataframe(result), width="stretch", hide_index=True)
    else:
        st.warning("선택 기간에 BB 진입 조건을 충족한 거래가 없습니다. 사이드바 BB 파라미터를 완화해 보세요.")
