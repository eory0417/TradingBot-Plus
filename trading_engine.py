"""기술적 지표 계산 + 가변형 시장 지정가 매매 엔진 (3단계).

두 가지 핵심 기능을 제공한다.

  1. 지표 계산
     ``pandas_ta`` 기반으로 대상 코인(BTC/ETH/SOL/XRP)의 15분봉 RSI(14)와
     ATR(14)를 실시간 계산하고, 최근 5개 종가의 선형 회귀 기울기(Slope)로 가격
     변동의 방향성과 강도를 측정한다(양수=상승, 음수=하락).

  2. 가변형 시장 지정가(Marketable Limit Order) 주문 엔진
     체결률을 높이고 슬리피지를 방지하기 위해 호가를 가로지르는 지정가 주문을
     IOC/FOK 조건으로 제출한다.
       * Long 진입: 현재 매도 호가(Ask)로 지정가 매수.
       * Short 진입: 현재 매수 호가(Bid)로 지정가 매도.
     미체결분이 발생하면 실패 사유를 구체적으로 로그에 남기고 즉시 취소한다.
     동시 보유 포지션은 최대 ``MAX_POSITIONS``(기본 2개)로 제한한다.

사용 예
-------
    import asyncio
    from exchange import create_exchange, close_exchange, load_markets_safe
    from trading_engine import TradingEngine

    async def main():
        ex = create_exchange()
        await load_markets_safe(ex)
        engine = TradingEngine(ex)
        ind = await engine.compute_indicators("BTC/USDT")
        print(ind)
        if engine.can_open():
            result = await engine.enter_position("BTC/USDT", "long")
            print(result)
        await close_exchange(ex)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import pandas_ta as ta

from config import settings
from derivatives_filter import DerivativesSnapshot
from logger import format_exception_detail, get_logger, log_exception

log = get_logger(__name__)

Side = Literal["long", "short"]

# 지표 계산에 필요한 룩백 길이.
RSI_LENGTH = 14
ATR_LENGTH = 14
SLOPE_WINDOW = 5  # 선형 회귀 기울기를 계산할 최근 캔들 수


# --------------------------------------------------------------------------- #
#  데이터 모델
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Indicators:
    """단일 심볼의 기술적 지표 스냅샷."""

    symbol: str
    timeframe: str
    last_price: float
    rsi: float
    atr: float
    slope: float          # 캔들당 가격 변화량(원자료 단위)
    slope_pct: float      # 마지막 가격 대비 정규화된 기울기(%) — 강도 측정용
    direction: str        # 'up' | 'down' | 'flat'


@dataclass(slots=True)
class OrderResult:
    """진입 주문 시도 결과."""

    symbol: str
    side: Side
    status: str           # 'filled' | 'partial' | 'unfilled' | 'rejected' | 'error'
    requested_amount: float
    filled_amount: float
    price: float
    order_id: str | None = None
    reason: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.status in ("filled", "partial") and self.filled_amount > 0


# --------------------------------------------------------------------------- #
#  지표 계산(순수 함수 — 네트워크 불필요, 테스트 용이)
# --------------------------------------------------------------------------- #
def linreg_slope(values: "pd.Series | np.ndarray", window: int = SLOPE_WINDOW) -> float:
    """최근 ``window``개 값의 선형 회귀 기울기를 반환한다.

    1차 최소제곱 적합(``numpy.polyfit``)을 사용한다. 양수는 상승, 음수는 하락
    추세를 의미하며, 절댓값이 클수록 변동 강도가 크다.
    """
    series = np.asarray(values, dtype=float)
    series = series[~np.isnan(series)]
    if series.size < 2:
        return 0.0
    recent = series[-window:]
    x = np.arange(recent.size, dtype=float)
    # polyfit는 [기울기, 절편]을 반환한다.
    slope = float(np.polyfit(x, recent, 1)[0])
    return slope


def compute_indicators_from_df(symbol: str, timeframe: str, df: pd.DataFrame) -> Indicators:
    """OHLCV 데이터프레임으로부터 RSI/ATR/기울기를 계산한다.

    ``df``는 ``high``, ``low``, ``close`` 컬럼을 포함해야 한다.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    rsi_series = ta.rsi(close, length=RSI_LENGTH)
    atr_series = ta.atr(high, low, close, length=ATR_LENGTH)

    rsi_val = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else float("nan")
    atr_val = float(atr_series.iloc[-1]) if atr_series is not None and not atr_series.empty else float("nan")

    slope = linreg_slope(close, SLOPE_WINDOW)
    last_price = float(close.iloc[-1])
    slope_pct = round(slope / last_price * 100, 4) if last_price else 0.0

    if slope > 0:
        direction = "up"
    elif slope < 0:
        direction = "down"
    else:
        direction = "flat"

    return Indicators(
        symbol=symbol,
        timeframe=timeframe,
        last_price=last_price,
        rsi=round(rsi_val, 2),
        atr=round(atr_val, 6),
        slope=round(slope, 6),
        slope_pct=slope_pct,
        direction=direction,
    )


# --------------------------------------------------------------------------- #
#  트레이딩 엔진
# --------------------------------------------------------------------------- #
class TradingEngine:
    """지표 계산 + 가변형 시장 지정가 주문 + 포지션 카운터."""

    def __init__(self, exchange, notifier=None) -> None:
        self.exchange = exchange
        self.notifier = notifier
        self.timeframe = settings.timeframe
        self.max_positions = settings.max_positions
        self.tif = settings.order_time_in_force  # 'IOC' 또는 'FOK'
        self.notional_usdt = settings.position_size_usdt
        self.leverage = settings.leverage
        self.margin_mode = settings.margin_mode  # 'isolated' 또는 'cross'
        # 현재 보유 포지션: {심볼: 방향}. 동시 진입 카운터의 단일 출처.
        self._open_positions: dict[str, Side] = {}
        # OI 변화율 계산용 캐시: {심볼: (open_interest, mono_ts)}
        self._oi_cache: dict[str, tuple[float, float]] = {}
        # 카운터 경쟁 상태를 막기 위한 락(여러 코루틴이 동시 진입 시도 가능).
        self._lock = asyncio.Lock()

    # ---- 포지션 카운터 ----
    @property
    def position_count(self) -> int:
        return len(self._open_positions)

    def can_open(self) -> bool:
        """동시 포지션 한도 내에서 신규 진입이 가능한지 여부."""
        return self.position_count < self.max_positions

    def register_exit(self, symbol: str) -> None:
        """포지션 청산 시 카운터에서 제거한다(상위 로직에서 호출)."""
        if self._open_positions.pop(symbol, None) is not None:
            log.info("Position closed | %s | open_now=%d", symbol, self.position_count)

    # ---- 지표 ----
    async def fetch_ohlcv_df(
        self,
        symbol: str,
        limit: int = 100,
        *,
        timeframe: str | None = None,
    ) -> pd.DataFrame | None:
        """OHLCV를 가져와 데이터프레임으로 반환한다."""
        tf = timeframe or self.timeframe
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="fetch_ohlcv", symbol=symbol, timeframe=tf)
            return None
        if not ohlcv:
            log.warning("No OHLCV returned | symbol=%s", symbol)
            return None
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        return df

    async def compute_indicators(self, symbol: str) -> Indicators | None:
        """심볼의 실시간 RSI/ATR/기울기 지표를 계산한다."""
        df = await self.fetch_ohlcv_df(symbol)
        if df is None or len(df) < max(RSI_LENGTH, ATR_LENGTH) + 1:
            log.warning("Insufficient candles for indicators | symbol=%s", symbol)
            return None
        indicators = compute_indicators_from_df(symbol, self.timeframe, df)
        log.debug(
            "Indicators | %s | price=%.4f RSI=%.2f ATR=%.4f slope=%.6f(%.3f%%) dir=%s",
            indicators.symbol,
            indicators.last_price,
            indicators.rsi,
            indicators.atr,
            indicators.slope,
            indicators.slope_pct,
            indicators.direction,
        )
        return indicators

    async def fetch_mtf_frames(
        self,
        symbol: str,
        timeframes: list[str],
        *,
        ema_len: int = 200,
    ) -> dict[str, pd.DataFrame]:
        """MTF 필터용 OHLCV 프레임 맵 (워밍업 여유 포함)."""
        out: dict[str, pd.DataFrame] = {}
        limit = max(ema_len + 20, 80)
        for tf in timeframes:
            df = await self.fetch_ohlcv_df(symbol, limit=limit, timeframe=tf)
            if df is not None and not df.empty:
                out[tf] = df
        return out

    # ---- 계정/시세 조회 ----
    async def fetch_balance_usdt(self) -> tuple[float, str | None]:
        """USDⓈ-M 선물 지갑의 USDT 가용 잔고를 조회한다.

        반환값: (잔고, 오류 메시지). 오류 시 잔고는 0.0이고 두 번째 값에 사유가 담긴다.
        """
        try:
            balance = await self.exchange.fetch_balance({"type": "future"})
            usdt = balance.get("USDT", {})
            free = float(usdt.get("free") or usdt.get("total") or 0.0)
            if free > 0:
                return free, None
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="fetch_balance")
            first_err = format_exception_detail(exc)
        else:
            first_err = None

        # fetch_balance 실패 또는 0일 때 fapi 잔고 API 직접 시도.
        try:
            raw = await self.exchange.fapiPrivateV2GetBalance()
            for item in raw:
                if str(item.get("asset", "")).upper() == "USDT":
                    free = float(item.get("availableBalance") or item.get("balance") or 0.0)
                    return free, None
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="fetch_balance_fapi")
            second_err = format_exception_detail(exc)
            if first_err:
                return 0.0, f"{first_err}\n--- fapi fallback ---\n{second_err}"
            return 0.0, second_err

        return 0.0, first_err

    async def fetch_mark_price(self, symbol: str) -> float | None:
        """심볼의 현재 마크/체결 가격을 조회한다."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close") or ticker.get("markPrice")
            return float(price) if price is not None else None
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="fetch_ticker", symbol=symbol)
            return None

    async def fetch_open_positions(self) -> list[dict]:
        """거래소의 현재 오픈 포지션(수량 != 0)을 조회한다."""
        try:
            positions = await self.exchange.fetch_positions()
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="fetch_positions")
            return []
        return [p for p in positions if abs(float(p.get("contracts") or 0)) > 0]

    async def get_position_contracts(self, symbol: str) -> tuple[Side | None, float]:
        """심볼의 거래소 잔여 수량 (contracts). 없으면 (None, 0)."""
        def _norm(s: str) -> str:
            return (
                str(s).upper()
                .replace(":USDT", "")
                .replace("/USDT", "")
                .replace("USDT", "")
                .replace("/", "")
                .strip()
            )

        target = _norm(symbol)
        for p in await self.fetch_open_positions():
            if _norm(str(p.get("symbol") or "")) != target:
                continue
            raw_side = (p.get("side") or "").lower()
            contracts = abs(float(p.get("contracts") or 0))
            if contracts <= 0 or raw_side not in ("long", "short"):
                continue
            return raw_side, contracts  # type: ignore[return-value]
        return None, 0.0

    async def flatten_symbol(self, symbol: str) -> OrderResult | None:
        """해당 심볼 거래소 잔여 포지션을 시장가(reduceOnly)로 전량 청산."""
        side, contracts = await self.get_position_contracts(symbol)
        if side is None or contracts <= 0:
            return None
        return await self.close_position(
            symbol, side, contracts, order_type="market", register_exit=True,
            verify_flat=False,
        )

    async def fetch_derivatives_snapshot(self, symbol: str) -> DerivativesSnapshot:
        """펀딩비·OI 스냅샷 (OI 변화율은 엔진 캐시 기준)."""
        fr_pct = 0.0
        oi = 0.0
        oi_change = 0.0
        ready = False
        details_parts: list[str] = []
        loop = asyncio.get_running_loop()

        try:
            fr_data = await self.exchange.fetch_funding_rate(symbol)
            fr_raw = float(fr_data.get("fundingRate") or 0.0)
            fr_pct = fr_raw * 100.0
            details_parts.append(f"fr={fr_pct:.4f}%")
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="fetch_funding_rate", symbol=symbol)
            details_parts.append("fr=unavailable")

        try:
            oi_data = await self.exchange.fetch_open_interest(symbol)
            oi = float(
                oi_data.get("openInterestAmount")
                or oi_data.get("openInterestValue")
                or oi_data.get("openInterest")
                or 0.0
            )
            prev = self._oi_cache.get(symbol)
            if prev and prev[0] > 0 and oi > 0:
                oi_change = (oi - prev[0]) / prev[0] * 100.0
            self._oi_cache[symbol] = (oi, loop.time())
            ready = oi > 0 or fr_pct != 0.0
            details_parts.append(f"oi={oi:.0f} Δ{oi_change:+.1f}%")
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="fetch_open_interest", symbol=symbol)
            details_parts.append("oi=unavailable")

        return DerivativesSnapshot(
            funding_rate_pct=fr_pct,
            open_interest=oi,
            oi_change_pct=oi_change,
            ready=ready,
            details=" ".join(details_parts),
        )

    async def submit_maker_limit(
        self,
        symbol: str,
        side: Side,
        *,
        leverage: int | None = None,
        notional: float,
        limit_price: float,
    ) -> OrderResult:
        """Post-only GTC 지정가 진입 (하이브리드 2차 체결용)."""
        await self._prepare_symbol(symbol, leverage=leverage)
        if limit_price <= 0 or notional <= 0:
            return OrderResult(
                symbol, side, "rejected", 0.0, 0.0, 0.0,
                reason="invalid maker params",
            )
        order_side = "buy" if side == "long" else "sell"
        raw_amount = notional / limit_price
        try:
            amount = float(self.exchange.amount_to_precision(symbol, raw_amount))
            price = float(self.exchange.price_to_precision(symbol, limit_price))
        except Exception:  # noqa: BLE001
            amount = round(raw_amount, 6)
            price = limit_price
        params = {"timeInForce": "GTC", "postOnly": True}
        try:
            order = await self.exchange.create_order(
                symbol, "limit", order_side, amount, price, params,
            )
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="maker_limit_entry", symbol=symbol, side=side)
            return OrderResult(
                symbol, side, "error", amount, 0.0, price,
                reason=f"{type(exc).__name__}: {exc}",
            )
        order_id = order.get("id")
        filled = float(order.get("filled") or 0.0)
        status = (order.get("status") or "").lower()
        if filled > 0:
            fill_price = float(order.get("average") or order.get("price") or price)
            st = "filled" if filled >= amount else "partial"
            return OrderResult(
                symbol, side, st, amount, filled, fill_price, order_id,
                "maker limit filled", order,
            )
        if status in ("open", "new"):
            return OrderResult(
                symbol, side, "unfilled", amount, 0.0, price, order_id,
                "maker limit resting", order,
            )
        return OrderResult(
            symbol, side, "unfilled", amount, 0.0, price, order_id,
            f"maker status={status}", order,
        )

    async def fetch_order_fill(
        self, symbol: str, order_id: str,
    ) -> OrderResult | None:
        """대기 중인 지정가 주문 체결 상태 조회."""
        if not order_id:
            return None
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="fetch_order", symbol=symbol, order_id=order_id)
            return None
        side_raw = (order.get("side") or "").lower()
        side: Side = "long" if side_raw == "buy" else "short"
        requested = float(order.get("amount") or 0.0)
        filled = float(order.get("filled") or 0.0)
        price = float(order.get("average") or order.get("price") or 0.0)
        status = (order.get("status") or "").lower()
        if status == "closed" or (filled > 0 and filled >= requested):
            st = "filled"
        elif filled > 0:
            st = "partial"
        elif status in ("canceled", "cancelled", "expired", "rejected"):
            st = "rejected"
        else:
            st = "unfilled"
        return OrderResult(
            symbol, side, st, requested, filled, price, order_id,
            f"order status={status}", order,
        )

    async def cancel_order_safe(self, symbol: str, order_id: str | None) -> None:
        """주문 취소(이미 종료된 경우 무시)."""
        if not order_id:
            return
        try:
            await self.exchange.cancel_order(order_id, symbol)
            log.info("Order canceled | symbol=%s id=%s", symbol, order_id)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "Cancel skipped | symbol=%s id=%s | %s: %s",
                symbol, order_id, type(exc).__name__, exc,
            )

    async def flatten_all(self) -> list[str]:
        """거래소의 모든 오픈 포지션을 시장가(reduceOnly)로 청산한다.

        시작 시 '봇이 추적하지 않는 잔여 포지션(수동 청산 가정)'을 정리하기 위해
        사용한다. 청산에 성공한 심볼 목록을 반환한다.
        """
        closed: list[str] = []
        for p in await self.fetch_open_positions():
            symbol = p.get("symbol")
            raw_side = (p.get("side") or "").lower()
            contracts = abs(float(p.get("contracts") or 0))
            if not symbol or raw_side not in ("long", "short") or contracts <= 0:
                continue
            result = await self.close_position(symbol, raw_side, contracts, order_type="market")  # type: ignore[arg-type]
            if result.is_filled:
                closed.append(symbol)
                log.info("Startup flatten | closed %s %s x%.6f", symbol, raw_side, contracts)
            else:
                log.warning("Startup flatten failed | %s | %s", symbol, result.reason)
        return closed

    # ---- 주문 ----
    async def _prepare_symbol(self, symbol: str, leverage: int | None = None) -> None:
        """주문 전 증거금 모드(격리/교차)와 레버리지를 설정한다."""
        lev = int(leverage) if leverage else self.leverage
        # 증거금 모드 설정(이미 동일 모드면 거래소가 에러를 내므로 무시).
        try:
            await self.exchange.set_margin_mode(self.margin_mode, symbol)
        except Exception as exc:  # noqa: BLE001 - 이미 설정된 경우 정상
            log.debug("set_margin_mode skipped | %s | %s: %s", symbol, type(exc).__name__, exc)
        try:
            await self.exchange.set_leverage(lev, symbol)
        except Exception as exc:  # noqa: BLE001 - 이미 설정돼 있으면 무시 가능
            log_exception(log, exc, context="set_leverage", symbol=symbol)

    async def _best_quote(self, symbol: str) -> tuple[float, float] | None:
        """호가창에서 (최우선 매수호가 bid, 최우선 매도호가 ask)를 반환한다."""
        try:
            book = await self.exchange.fetch_order_book(symbol, limit=5)
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="fetch_order_book", symbol=symbol)
            return None
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if not bids or not asks:
            log.warning("Empty order book | symbol=%s", symbol)
            return None
        return float(bids[0][0]), float(asks[0][0])

    async def enter_position(
        self,
        symbol: str,
        side: Side,
        *,
        leverage: int | None = None,
        notional: float | None = None,
    ) -> OrderResult:
        """가변형 시장 지정가 방식으로 Long/Short 진입을 시도한다.

        동시 포지션 한도를 먼저 확인하고, 호가를 가로지르는 IOC/FOK 지정가
        주문을 제출한다. 미체결분이 남으면 사유를 로깅하고 즉시 취소한다.
        체결에 성공하면 포지션 카운터를 증가시킨다.

        ``leverage`` / ``notional`` 을 주면 해당 진입에만 적용한다(실시간 설정·자동
        레버리지 반영). 생략 시 엔진 기본값(설정 스냅샷)을 사용한다.
        """
        async with self._lock:
            # ---- 동시 포지션 카운터 검사 ----
            if not self.can_open():
                reason = (
                    f"max position limit reached ({self.position_count}/{self.max_positions})"
                )
                log.warning("Entry blocked | %s %s | %s", symbol, side, reason)
                return OrderResult(symbol, side, "rejected", 0.0, 0.0, 0.0, reason=reason)

            if symbol in self._open_positions:
                reason = f"position already open on {symbol} ({self._open_positions[symbol]})"
                log.warning("Entry blocked | %s %s | %s", symbol, side, reason)
                return OrderResult(symbol, side, "rejected", 0.0, 0.0, 0.0, reason=reason)

            # 한도 내 슬롯을 선점하기 위해 카운터에 임시 등록(체결 실패 시 롤백).
            self._open_positions[symbol] = side

        retries = max(0, int(settings.entry_retry_count))
        delay_s = max(0, int(settings.entry_retry_delay_ms)) / 1000.0
        result = OrderResult(symbol, side, "unfilled", 0.0, 0.0, 0.0, reason="no attempt")
        for attempt in range(retries + 1):
            chase = float(settings.entry_chase_bps) * attempt
            result = await self._submit_marketable_limit(
                symbol, side, leverage=leverage, notional=notional, chase_bps=chase,
            )
            if result.is_filled:
                break
            if attempt < retries and delay_s > 0:
                await asyncio.sleep(delay_s)

        if not result.is_filled and settings.entry_fallback_market:
            log.warning(
                "IOC entry failed → market fallback | %s %s | last=%s",
                symbol, side, result.reason,
            )
            result = await self._market_entry(
                symbol, side, leverage=leverage, notional=notional,
            )

        if not result.is_filled:
            async with self._lock:
                self._open_positions.pop(symbol, None)
        else:
            log.info(
                "Position opened | %s %s | filled=%.6f | open_now=%d",
                symbol, side, result.filled_amount, self.position_count,
            )
            if self.notifier is not None:
                await self.notifier.send_trade(
                    f"ENTRY {side.upper()}",
                    symbol,
                    f"filled={result.filled_amount} @ {result.price}",
                )
        return result

    async def increase_position(
        self,
        symbol: str,
        side: Side,
        *,
        leverage: int | None = None,
        notional: float | None = None,
    ) -> OrderResult:
        """기존 포지션에 같은 방향으로 추가 진입(피라미딩)한다.

        동시 포지션 카운터는 이미 해당 심볼이 점유 중이므로 건드리지 않고,
        주문만 추가로 제출한다. 레버리지를 주면 추가 분에 맞춰 재설정한다.
        """
        result = await self._submit_marketable_limit(
            symbol, side, leverage=leverage, notional=notional,
        )
        if not result.is_filled and settings.entry_fallback_market:
            result = await self._market_entry(
                symbol, side, leverage=leverage, notional=notional,
            )
        if result.is_filled:
            log.info(
                "Position increased | %s %s | add_filled=%.6f @ %.4f",
                symbol, side, result.filled_amount, result.price,
            )
        return result

    async def _market_entry(
        self,
        symbol: str,
        side: Side,
        *,
        leverage: int | None = None,
        notional: float | None = None,
    ) -> OrderResult:
        """시장가 진입 (IOC 실패 시 폴백용)."""
        await self._prepare_symbol(symbol, leverage=leverage)
        notional_usdt = float(notional) if notional else self.notional_usdt
        quote = await self._best_quote(symbol)
        if quote is None:
            return OrderResult(symbol, side, "error", 0.0, 0.0, 0.0, reason="no order book")
        best_bid, best_ask = quote
        ref = best_ask if side == "long" else best_bid
        if ref <= 0:
            return OrderResult(symbol, side, "error", 0.0, 0.0, 0.0, reason="bad quote")
        order_side = "buy" if side == "long" else "sell"
        raw_amount = notional_usdt / ref
        try:
            amount = float(self.exchange.amount_to_precision(symbol, raw_amount))
        except Exception:  # noqa: BLE001
            amount = round(raw_amount, 6)
        try:
            order = await self.exchange.create_order(
                symbol, "market", order_side, amount, None, {},
            )
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="market_entry", symbol=symbol, side=side)
            return OrderResult(
                symbol, side, "error", amount, 0.0, ref,
                reason=f"{type(exc).__name__}: {exc}",
            )
        filled = float(order.get("filled") or 0.0)
        fill_price = float(order.get("average") or order.get("price") or ref)
        if filled > 0:
            return OrderResult(
                symbol, side, "filled", amount, filled, fill_price,
                order.get("id"), "market fallback", order,
            )
        return OrderResult(
            symbol, side, "unfilled", amount, 0.0, fill_price,
            order.get("id"), "market unfilled", order,
        )

    async def _submit_marketable_limit(
        self,
        symbol: str,
        side: Side,
        *,
        leverage: int | None = None,
        notional: float | None = None,
        chase_bps: float = 0.0,
    ) -> OrderResult:
        """호가를 가로지르는 지정가(IOC/FOK) 주문을 제출하고 결과를 평가한다."""
        await self._prepare_symbol(symbol, leverage=leverage)
        notional_usdt = float(notional) if notional else self.notional_usdt

        quote = await self._best_quote(symbol)
        if quote is None:
            return OrderResult(symbol, side, "error", 0.0, 0.0, 0.0, reason="no order book")
        best_bid, best_ask = quote

        # Long 진입은 매도호가(Ask)로 매수, Short 진입은 매수호가(Bid)로 매도하여
        # 호가를 가로질러 즉시 체결을 노린다(= 가변형 시장 지정가).
        if side == "long":
            order_side = "buy"
            limit_price = best_ask * (1 + chase_bps / 10_000.0) if chase_bps else best_ask
        else:
            order_side = "sell"
            limit_price = best_bid * (1 - chase_bps / 10_000.0) if chase_bps else best_bid

        # 명목 가치(USDT)를 수량으로 환산하고 거래소 정밀도에 맞춰 반올림한다.
        raw_amount = notional_usdt / limit_price
        try:
            amount = float(self.exchange.amount_to_precision(symbol, raw_amount))
            price = float(self.exchange.price_to_precision(symbol, limit_price))
        except Exception:  # noqa: BLE001 - 정밀도 헬퍼 실패 시 원값 사용
            amount = round(raw_amount, 6)
            price = limit_price

        params = {"timeInForce": self.tif}
        log.info(
            "Submitting marketable-limit | %s %s | price=%s amount=%s tif=%s chase_bps=%s",
            symbol, order_side, price, amount, self.tif, chase_bps,
        )

        try:
            order = await self.exchange.create_order(
                symbol, "limit", order_side, amount, price, params
            )
        except Exception as exc:  # noqa: BLE001 - 진입 실패는 표준 형식으로 기록
            log_exception(
                log, exc, context="entry_order",
                symbol=symbol, side=side, price=price, amount=amount,
            )
            return OrderResult(
                symbol, side, "error", amount, 0.0, price,
                reason=f"{type(exc).__name__}: {exc}",
            )

        return await self._evaluate_and_cleanup(symbol, side, order, amount, price)

    async def _evaluate_and_cleanup(
        self, symbol: str, side: Side, order: dict, requested: float, price: float
    ) -> OrderResult:
        """주문 체결 상태를 평가하고, 미체결분은 사유를 남기고 즉시 취소한다."""
        order_id = order.get("id")
        status = (order.get("status") or "").lower()
        filled = float(order.get("filled") or 0.0)

        # 완전 체결.
        if status == "closed" or filled >= requested > 0:
            log.info("Order fully filled | %s | id=%s filled=%.6f", symbol, order_id, filled)
            return OrderResult(symbol, side, "filled", requested, filled, price, order_id, "filled", order)

        # 부분 체결: 일부만 체결되고 IOC로 나머지는 자동 취소됨.
        if 0 < filled < requested:
            reason = f"partial fill: {filled}/{requested} (remainder canceled by {self.tif})"
            log.warning("Order partially filled | %s | id=%s | %s", symbol, order_id, reason)
            await self._cancel_if_open(symbol, order_id)
            return OrderResult(symbol, side, "partial", requested, filled, price, order_id, reason, order)

        # 미체결: 사유를 구체적으로 로깅하고 즉시 취소 처리한다.
        reason = (
            f"unfilled (status={status or 'unknown'}, filled=0) - likely no liquidity "
            f"at {price} or {self.tif} could not match"
        )
        log.error(
            "ENTRY UNFILLED | symbol=%s side=%s id=%s price=%s amount=%s | %s",
            symbol, side, order_id, price, requested, reason,
        )
        await self._cancel_if_open(symbol, order_id)
        if self.notifier is not None:
            await self.notifier.send_error("entry_unfilled", f"{symbol} {side}: {reason}")
        return OrderResult(symbol, side, "unfilled", requested, 0.0, price, order_id, reason, order)

    async def _cancel_if_open(self, symbol: str, order_id: str | None) -> None:
        """주문이 아직 열려 있으면 즉시 취소한다(IOC/FOK는 보통 자동 취소됨)."""
        if not order_id:
            return
        try:
            await self.exchange.cancel_order(order_id, symbol)
            log.info("Order canceled | symbol=%s id=%s", symbol, order_id)
        except Exception as exc:  # noqa: BLE001 - 이미 취소/체결된 경우 정상
            # OrderNotFound 등은 이미 종료된 주문이므로 디버그 수준으로 기록.
            log.debug(
                "Cancel skipped (already closed?) | symbol=%s id=%s | %s: %s",
                symbol, order_id, type(exc).__name__, exc,
            )

    # ---- 청산 ----
    async def close_position(
        self,
        symbol: str,
        side: Side,
        amount: float,
        order_type: str = "market",
        *,
        register_exit: bool = True,
        verify_flat: bool = True,
    ) -> OrderResult:
        """오픈 포지션을 청산한다.

        ``order_type='market'``  : 즉시 시장가 청산(고정 손절/트레일링 스톱용).
        ``order_type='marketable_limit'`` : 호가를 가로지르는 지정가 청산(시간 청산용).
        ``register_exit=False`` : 부분 익절 시 포지션 카운터를 유지한다.
        ``verify_flat`` : 전량 청산 후 거래소 잔량(dust)이 있으면 시장가로 한 번 더 정리.

        Long 포지션은 매도(sell)로, Short 포지션은 매수(buy)로 반대 방향 주문을
        ``reduceOnly``로 제출한다.
        """
        close_side = "sell" if side == "long" else "buy"
        # 전량 청산 시 봇 추적 수량보다 거래소 잔량이 크면 거래소 수량으로 맞춤
        # (부분익절·정밀도 반올림으로 dust 가 남는 케이스 방지).
        close_amount = float(amount)
        if register_exit and verify_flat:
            ex_side, ex_amt = await self.get_position_contracts(symbol)
            if ex_side == side and ex_amt > close_amount:
                log.info(
                    "Close size from exchange | %s | bot=%.8f exchange=%.8f",
                    symbol, close_amount, ex_amt,
                )
                close_amount = ex_amt

        try:
            if order_type == "marketable_limit":
                quote = await self._best_quote(symbol)
                if quote is None:
                    raise RuntimeError("no order book for marketable-limit close")
                best_bid, best_ask = quote
                # 청산도 호가를 가로질러 즉시 체결을 노린다.
                price = best_bid if close_side == "sell" else best_ask
                price = float(self.exchange.price_to_precision(symbol, price))
                amt = float(self.exchange.amount_to_precision(symbol, close_amount))
                if amt <= 0:
                    raise RuntimeError(
                        f"close amount rounds to 0 (raw={close_amount})"
                    )
                order = await self.exchange.create_order(
                    symbol, "limit", close_side, amt, price,
                    {"timeInForce": self.tif, "reduceOnly": True},
                )
            else:
                amt = float(self.exchange.amount_to_precision(symbol, close_amount))
                if amt <= 0:
                    raise RuntimeError(
                        f"close amount rounds to 0 (raw={close_amount})"
                    )
                price = 0.0
                order = await self.exchange.create_order(
                    symbol, "market", close_side, amt, None, {"reduceOnly": True},
                )
        except Exception as exc:  # noqa: BLE001 - 청산 실패는 표준 형식으로 기록
            log_exception(
                log, exc, context="close_order",
                symbol=symbol, side=side, amount=close_amount, order_type=order_type,
            )
            return OrderResult(
                symbol, side, "error", close_amount, 0.0, 0.0,
                reason=f"{type(exc).__name__}: {exc}",
            )

        filled = float(order.get("filled") or 0.0)
        fill_price = float(order.get("average") or order.get("price") or price or 0.0)
        # 체결량이 비어 있으면 잔량 조회로 성공 여부 판정 (오판 filled=요청량 금지).
        if filled <= 0 and register_exit and verify_flat:
            await asyncio.sleep(0.3)
            rem_side, rem_amt = await self.get_position_contracts(symbol)
            if rem_side is None or rem_amt <= 0:
                filled = amt
            else:
                closed_est = max(0.0, close_amount - rem_amt)
                if closed_est > 0:
                    filled = closed_est

        if filled <= 0:
            log.error(
                "CLOSE UNFILLED | %s %s | id=%s amount=%s",
                symbol, side, order.get("id"), amt,
            )
            return OrderResult(
                symbol, side, "unfilled", close_amount, 0.0, fill_price,
                order.get("id"), "close unfilled", order,
            )

        # 전량 청산 후 dust 잔량 정리 (정밀도 반올림으로 남는 소량).
        if register_exit and verify_flat:
            rem_side, rem_amt = await self.get_position_contracts(symbol)
            if rem_side == side and rem_amt > 0:
                try:
                    dust_amt = float(self.exchange.amount_to_precision(symbol, rem_amt))
                except Exception:  # noqa: BLE001
                    dust_amt = rem_amt
                if dust_amt > 0:
                    log.warning(
                        "Residual dust after close | %s %s rem=%.8f → market flatten",
                        symbol, side, rem_amt,
                    )
                    try:
                        dust_order = await self.exchange.create_order(
                            symbol, "market", close_side, dust_amt, None,
                            {"reduceOnly": True},
                        )
                        dust_filled = float(dust_order.get("filled") or dust_amt)
                        filled += dust_filled
                        dust_px = float(
                            dust_order.get("average")
                            or dust_order.get("price")
                            or fill_price
                            or 0.0
                        )
                        if dust_px > 0:
                            fill_price = dust_px
                    except Exception as exc:  # noqa: BLE001
                        log_exception(
                            log, exc, context="dust_flatten",
                            symbol=symbol, rem=rem_amt,
                        )

        if register_exit:
            self.register_exit(symbol)
        log.info(
            "Position closed | %s %s | type=%s filled=%.6f price=%.4f | register_exit=%s | open_now=%d",
            symbol, side, order_type, filled, fill_price, register_exit, self.position_count,
        )
        return OrderResult(
            symbol, side, "filled", close_amount, filled, fill_price,
            order.get("id"), f"closed via {order_type}", order,
        )
