"""핵심 트레이딩 루프 오케스트레이션 (4단계).

뉴스 분석(2단계) + 기술적 지표/주문 엔진(3단계) + 동적 익절/손절 전략(4단계)을
하나의 비동기 루프로 묶어 실행하고, 결과를 공유 상태(:mod:`state`)에 기록하여
Streamlit GUI가 실시간으로 표시할 수 있게 한다.

동작 모드
---------
  * LIVE 모드: 실제 바이낸스 자격증명이 있으면 ccxt로 시세/주문을 처리한다.
  * SIM 모드 : 자격증명이 플레이스홀더이면 합성 가격(랜덤워크)으로 페이퍼
    트레이딩을 수행한다. 단, 뉴스 파이프라인(무료 RSS + FinBERT)은 두 모드
    모두에서 실제로 동작하므로 GUI를 그대로 시연할 수 있다.

진입 규칙(뉴스 트레이딩)
------------------------
강한 감성 뉴스(|score| >= 임계값)가 특정 코인을 언급하고, 지표가 방향을
확인(긍정+기울기>0 → Long, 부정+기울기<0 → Short)하면 가변형 시장 지정가로
진입한다. 청산은 :class:`strategy.Position`의 동적 익절/손절 규칙을 따른다.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import finetune
from bb_breakout import BB_OHLCV_LIMIT, evaluate_bb_entry
from config import settings
from logger import format_exception_brief, get_logger, log_exception
from derivatives_filter import gate_entry as deriv_gate_entry
from mtf_filter import evaluate_mtf_from_frames, gate_entry, parse_mtf_tfs
from news_analyzer import AnalyzedNews, NewsAnalyzer
from news_category import category_exit_overrides, classify_news
from news_weights import effective_news_score
from pending_entry import PendingEntry
from sizing import SizingState, compute_notional
from state import STATE, NewsView, PositionView, ClosedChartView
from strategy import ExitSignal, Position, Side
from trade_log import record_trade
from symbol_util import normalize_symbol_key
from kst_util import format_kst
from trading_engine import TradingEngine, compute_indicators_from_df, RSI_LENGTH, ATR_LENGTH
from translator import translate_to_korean

log = get_logger(__name__)


def score_to_leverage(score: float) -> int:
    """뉴스 점수 절대값 → 레버리지 배율(0.7→1x, 0.8→2x, 0.9→3x, 1.0→4x)."""
    a = abs(score)
    if a >= 0.9995:
        return 4
    if a >= 0.9:
        return 3
    if a >= 0.8:
        return 2
    return 1


def auto_leverage_decision(score: float, slope: float) -> tuple[Side | None, int]:
    """뉴스 점수 강도에 따른 (방향, 레버리지) 결정(자동 레버리지 모드).

    반환값 ``(side, leverage)``. ``side`` 가 ``None`` 이면 진입하지 않는다.

    절대값 기준 레버리지: 0.7~ → 1배, 0.8~ → 2배, 0.9~ → 3배, 1.0 → 4배.
    추세 일치(점수>0 & 기울기>0, 점수<0 & 기울기<0)면 위 배율로 진입한다.
    추세가 역방향이라도 ``|점수| == 1.0``(최대 확신)이면 추세 필터를 무시하고
    **2배**로 진입한다. 그 외 역방향은 진입하지 않는다.
    """
    a = abs(score)
    if a < settings.news_score_threshold:
        return None, 0
    extreme = a >= 0.9995  # 사실상 1.0
    base = score_to_leverage(score)

    if score > 0:
        if slope > 0:
            return "long", base
        return ("long", 2) if extreme else (None, 0)
    if score < 0:
        if slope < 0:
            return "short", base
        return ("short", 2) if extreme else (None, 0)
    return None, 0

# 코인 심볼 <-> 뉴스 키워드 매핑(뉴스 제목에서 대상 코인 탐지).
# 영어(RSS·번역본) + 한국어(coinnesskr 번역 실패 시 fallback) 키워드를 함께 둔다.
SYMBOL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "BTC/USDT": ("bitcoin", "btc", "비트코인"),
    "ETH/USDT": ("ethereum", "ether", "eth", "이더리움", "이더"),
    "SOL/USDT": ("solana", "sol", "솔라나"),
    "XRP/USDT": ("ripple", "xrp", "리플"),
}

# SIM 모드 기준 시작 가격.
SIM_BASE_PRICES: dict[str, float] = {
    "BTC/USDT": 65000.0,
    "ETH/USDT": 3500.0,
    "SOL/USDT": 150.0,
    "XRP/USDT": 0.60,
}

BB_FAIL_LABELS: dict[str, str] = {
    "BB_WIDTH": "밴드 폭 부족(횡보)",
    "VOLUME": "거래량 부족",
    "TREND": "추세 필터 미달",
    "RANGE": "캔들 변동폭 부족",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def detect_symbols(title: str, universe: list[str], title_ko: str = "") -> list[str]:
    """뉴스 제목에서 언급된 대상 코인 심볼을 추출한다.

    영어 제목과(있으면) 한국어 원문을 함께 검사해, 번역이 부정확하더라도
    한국어 코인명으로 매칭할 수 있게 한다.
    """
    text = f"{title} {title_ko}".lower()
    hits = []
    for symbol in universe:
        for kw in SYMBOL_KEYWORDS.get(symbol, ()):  # 키워드 매칭
            if kw in text:
                hits.append(symbol)
                break
    return hits


def has_real_credentials() -> bool:
    """자격증명이 실제처럼 보이는지(플레이스홀더가 아닌지) 판정."""
    key = settings.binance_api_key.get_secret_value()
    return bool(key) and "your_" not in key


def exchange_mode_label() -> str:
    """현재 거래소 연결 모드 라벨(SIM / DEMO / LIVE)."""
    if not has_real_credentials():
        return "SIM(페이퍼)"
    if settings.binance_testnet:
        return "DEMO"
    return "LIVE"


# --------------------------------------------------------------------------- #
#  SIM 모드용 합성 시장(페이퍼 트레이딩)
# --------------------------------------------------------------------------- #
class SimMarket:
    """랜덤워크 기반 합성 OHLCV/잔고/체결을 제공하는 모의 시장."""

    def __init__(self, symbols: list[str]) -> None:
        self.balance = 10_000.0
        self._closes: dict[str, deque] = {}
        self._volumes: dict[str, deque] = {}
        rng = np.random.default_rng(42)
        for sym in symbols:
            base = SIM_BASE_PRICES.get(sym, 100.0)
            # 초기 120개 캔들을 약한 추세 + 노이즈로 시드.
            drift = rng.normal(0, base * 0.0008, 120).cumsum()
            series = base + drift + rng.normal(0, base * 0.0005, 120)
            self._closes[sym] = deque(series.tolist(), maxlen=400)
            self._volumes[sym] = deque(
                rng.uniform(0.5, 3.0, 120).tolist(), maxlen=400,
            )
        self._rng = rng

    def tick(self, symbol: str) -> float:
        """가격을 한 스텝 진행시키고 새 종가를 반환한다."""
        closes = self._closes[symbol]
        last = closes[-1]
        nxt = max(last * (1 + self._rng.normal(0, 0.0015)), 1e-6)
        closes.append(nxt)
        self._volumes[symbol].append(float(self._rng.uniform(0.5, 3.0)))
        return nxt

    def price(self, symbol: str) -> float:
        return self._closes[symbol][-1]

    def ohlcv(self, symbol: str, limit: int = 120) -> list[list[float]]:
        closes = list(self._closes[symbol])[-limit:]
        volumes = list(self._volumes[symbol])[-limit:]
        now_ms = int(_now().timestamp() * 1000)
        rows = []
        for i, c in enumerate(closes):
            jitter = abs(c) * 0.001
            ts = now_ms - (len(closes) - i) * 60_000
            vol = volumes[i] if i < len(volumes) else 1.0
            rows.append([ts, c - jitter, c + jitter, c - jitter, c, vol])
        return rows


# --------------------------------------------------------------------------- #
#  트레이딩 봇
# --------------------------------------------------------------------------- #
class TradingBot:
    """뉴스 + 지표 + 전략을 묶는 핵심 오케스트레이터."""

    def __init__(self, state=STATE) -> None:
        self.state = state
        self.symbols = settings.symbols
        self.sim = not has_real_credentials()
        # 진입 파라미터(명목금액/레버리지/임계값/모니터 주기)는 캐시하지 않고
        # 사용 시점에 settings 에서 직접 읽어 '실시간 설정 반영'을 지원한다.
        # 단, 이미 진입한 포지션은 진입 시 스냅샷한 값을 그대로 유지한다.

        # 수동 재학습 트리거(GUI 버튼 → _finetune_loop 가 감지).
        self._finetune_now = False

        self.positions: dict[str, Position] = {}
        # 심볼별 최신 뉴스 컨텍스트(점수/내용).
        self._latest_news: dict[str, tuple[str, float]] = defaultdict(lambda: ("", 0.0))
        self._lock = asyncio.Lock()
        self._running = False
        self._started_at: datetime | None = None
        self._ohlcv_cache: dict[str, list[list[float]]] = {}
        self._bb_last_bar_ts: dict[str, int] = {}
        self._last_balance_fetch: float = 0.0
        self._balance_poll_sec = 60.0
        self._sizing = SizingState()
        self._mtf_cache: dict[str, tuple[float, object]] = {}  # symbol -> (mono_ts, MtfTrend)
        self._deriv_cache: dict[str, tuple[float, object]] = {}  # symbol -> (mono_ts, DerivativesSnapshot)
        self._pending_entries: dict[str, PendingEntry] = {}
        self._entry_inflight: set[str] = set()  # 주문 중 고아정리 레이스 방지
        self._last_orphan_check: float = 0.0
        self._orphan_poll_sec = 60.0

        # 모드별 구성.
        self.exchange = None
        self.engine: TradingEngine | None = None
        self.notifier = None
        self.sim_market: SimMarket | None = SimMarket(self.symbols) if self.sim else None
        self.news = NewsAnalyzer()

    # ---- 라이프사이클 ----
    async def run(self) -> None:
        """봇을 시작하고 뉴스 태스크 + 모니터 루프를 병행 실행한다."""
        self._started_at = _now()
        self.state.clear_news()
        self._running = True
        mode = exchange_mode_label()
        self.state.set_running(True, status=f"running ({mode})")
        self._emit_log("INFO", "system", f"트레이딩 봇 시작 | 모드={mode} | 심볼={self.symbols}")
        log.info("TradingBot starting | mode=%s | symbols=%s", mode, self.symbols)

        try:
            self._emit_log("INFO", "system", "초기화 중 — 거래소/잔고 연결…")
            await self._setup()
            self._emit_log("INFO", "system", "초기화 완료 — 뉴스·차트 수집 시작 (FinBERT 로딩 중일 수 있음)")
            self._emit_log(
                "INFO", "system",
                f"뉴스 워밍업: 최근 기사는 화면 표시 · 진입은 시작 후 "
                f"{settings.news_entry_grace_seconds}초 + 발행 "
                f"{settings.news_max_age_minutes:.0f}분 이내만",
            )
            news_task = asyncio.create_task(
                self.news.start(
                    self._on_news,
                    on_status=lambda msg: self._emit_log("INFO", "system", msg),
                )
            )
            monitor_task = asyncio.create_task(self._monitor_loop())
            finetune_task = asyncio.create_task(self._finetune_loop())
            await asyncio.gather(news_task, monitor_task, finetune_task)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="bot_run")
            self._emit_log("ERROR", "system", f"봇 루프 오류: {exc}")
        finally:
            await self._teardown()
            self.state.set_running(False, status="stopped")
            self._emit_log("INFO", "system", "봇 종료")

    def stop(self) -> None:
        self._running = False
        self.news.stop()
        self.positions.clear()
        self._pending_entries.clear()
        self.state.clear_positions()
        self.state.pop_close_requests()
        self.state.set_running(False, status="stopped")

    async def _setup(self) -> None:
        if self.sim:
            # 클린 스타트: 이전 세션의 잔여 포지션 표시를 비우고 무포지션으로 시작.
            self.positions.clear()
            self.state.clear_positions()
            self.state.clear_closed_charts()
            self.state.set_balance(self.sim_market.balance)
            self._emit_log("INFO", "system", "SIM 모드: 합성 시장 + 페이퍼 잔고 10,000 USDT (무포지션 시작)")
            return
        # LIVE 모드: 실제 거래소/알림 구성.
        from exchange import create_exchange, diagnose_exchange, load_markets_safe
        from notifier import TelegramNotifier

        self.exchange = create_exchange()
        try:
            diag_lines = await diagnose_exchange(self.exchange)
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="diagnose_exchange")
            diag_lines = [f"진단 중단: {format_exception_brief(exc)}"]
        for line in diag_lines:
            log.info("연결 진단 | %s", line)
            self._emit_log("INFO", "system", f"연결 진단: {line}")

        markets_ok, market_err = await load_markets_safe(self.exchange)
        if not markets_ok:
            self._emit_log("ERROR", "system", f"마켓 로드 실패:\n{market_err or 'unknown'}")
        self.notifier = TelegramNotifier()
        self.engine = TradingEngine(self.exchange, notifier=self.notifier)

        # ---- 시작 시 잔여 포지션 정리(클린 스타트) ----
        # entry 만 있고 exit 가 없는(봇이 추적하지 않는) 거래소 잔여 포지션은
        # 외부에서 수동 청산된 것으로 간주하고 시장가로 정리해 무포지션으로 시작한다.
        try:
            closed = await self.engine.flatten_all()
            if closed:
                self._emit_log(
                    "WARNING", "system",
                    f"시작 정리: 잔여 오픈 포지션 청산({len(closed)}개) → {', '.join(closed)}",
                )
            else:
                self._emit_log("INFO", "system", "시작 정리: 잔여 오픈 포지션 없음 (무포지션 시작)")
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="startup_flatten")
            self._emit_log("WARNING", "system", f"시작 정리 중 경고: {exc}")
        self.positions.clear()
        self.state.clear_positions()
        self.state.clear_closed_charts()

        bal, bal_err = await self.engine.fetch_balance_usdt()
        self.state.set_balance(bal)
        if bal_err:
            self._emit_log("ERROR", "system", f"잔고 조회 실패:\n{bal_err}")
        elif exchange_mode_label() == "DEMO" and bal == 0.0:
            self._emit_log(
                "WARNING", "system",
                "Demo 잔고가 0입니다. demo.binance.com API 키인지 확인하세요.",
            )

    async def _teardown(self) -> None:
        try:
            if not self.sim and self.engine is not None:
                for sym, pending in list(self._pending_entries.items()):
                    await self.engine.cancel_order_safe(sym, pending.order_id)
            self._pending_entries.clear()
            if self.exchange is not None:
                from exchange import close_exchange
                await close_exchange(self.exchange)
            if self.notifier is not None:
                await self.notifier.close()
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="bot_teardown")

    # ---- 월간 자동 파인튜닝(재학습) 루프 ----
    async def _finetune_loop(self) -> None:
        """주기(기본 30일)마다 또는 수동 트리거 시 FinBERT를 재학습한다.

        학습은 별도 스레드에서 수행하며(블로킹 방지), 성공 시 감성 모델을
        새 가중치로 핫스왑한다. 실패는 흡수되어 트레이딩 루프를 막지 않는다.
        """
        if not settings.finetune_enabled:
            self._emit_log("INFO", "system", "자동 재학습 비활성화됨 (FINETUNE_ENABLED=false)")
            return
        # 시작 직후 폭주 방지를 위해 잠시 대기.
        await asyncio.sleep(10)
        while self._running:
            try:
                manual = self._finetune_now
                if manual:
                    self._finetune_now = False
                if manual or finetune.due_for_run():
                    trigger = "수동" if manual else "월간"
                    self._emit_log(
                        "INFO", "system",
                        f"FinBERT 재학습 시작({trigger}) — 누적 샘플 {finetune.sample_count()}건",
                    )
                    ok = await asyncio.to_thread(finetune.run_finetune)
                    if ok:
                        await asyncio.to_thread(self.news.sentiment.reload)
                        self._emit_log("INFO", "system", "FinBERT 재학습 완료 — 새 모델 적용됨")
                    else:
                        self._emit_log(
                            "WARNING", "system",
                            "FinBERT 재학습 건너뜀(샘플 부족 또는 오류) — 로그 확인",
                        )
            except Exception as exc:  # noqa: BLE001 - 루프 생존
                log_exception(log, exc, context="finetune_loop")
            # 30초 간격으로 트리거/주기 확인(정지 시 빠르게 빠져나옴).
            await asyncio.sleep(30)

    def trigger_finetune(self) -> None:
        """GUI 등 외부에서 즉시 재학습을 요청한다(다음 루프 틱에 실행)."""
        self._finetune_now = True

    def _resolve_news_titles(self, item: AnalyzedNews) -> tuple[str, str]:
        """표시·로그·알림용 (영문, 한글) 제목 쌍을 반환한다."""
        en = item.title
        if item.item.origin == "coinnesskr" and item.item.title_ko:
            ko = item.item.title_ko
        else:
            ko = translate_to_korean(en)
        return en, ko

    @staticmethod
    def _format_bilingual(
        en: str, ko: str, score: float | None = None, *, max_len: int = 120,
    ) -> str:
        """로그·알림용 영문+한글 한 줄 요약."""
        en_show = en if len(en) <= max_len else en[: max_len - 1] + "…"
        ko_show = (ko or "").strip()
        if not ko_show:
            ko_show = "번역 없음"
        elif len(ko_show) > max_len:
            ko_show = ko_show[: max_len - 1] + "…"
        if score is not None:
            return f"뉴스({score:+.2f}) | EN: {en_show} | 한글: {ko_show}"
        return f"EN: {en_show} | 한글: {ko_show}"

    def _news_context(
        self, news: str, score: float | None = None, news_ko: str | None = None,
    ) -> str:
        """로그용 뉴스 요약(영문 원문 + 한글 번역)."""
        ko = news_ko if news_ko is not None else translate_to_korean(news)
        return self._format_bilingual(news, ko, score)

    def _news_entry_allowed(self) -> bool:
        """시작 후 grace 기간이 지났는지(진입 허용 여부)."""
        if self._started_at is None:
            return False
        elapsed = (_now() - self._started_at).total_seconds()
        return elapsed >= settings.news_entry_grace_seconds

    def _is_fresh_entry_news(self, item: AnalyzedNews) -> bool:
        """봇 시작·grace 이후 발행된, 너무 오래되지 않은 기사만 진입 허용."""
        if self._started_at is None:
            return False
        pub = item.item.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        cutoff = self._started_at
        if pub < cutoff:
            return False
        max_age = settings.news_max_age_minutes
        if max_age > 0 and (_now() - pub).total_seconds() > max_age * 60:
            return False
        return True

    # ---- 뉴스 콜백(진입 트리거) ----
    async def _on_news(self, item: AnalyzedNews) -> None:
        triggered_at = _now()
        pub = item.item.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        title_en, title_ko = self._resolve_news_titles(item)
        self.state.add_news(
            NewsView(
                time=format_kst(pub, "%H:%M:%S"),
                title=title_en,
                score=item.score,
                label=item.label,
                source=item.item.source,
                title_ko=title_ko,
                at_ms=int(triggered_at.timestamp() * 1000),
                published_at_ms=int(pub.timestamp() * 1000),
            )
        )
        # 월간 재학습용 샘플 누적(현재 모델이 부여한 감성 라벨을 정답으로 기록).
        try:
            finetune.record_sample(item.title, item.label)
        except Exception as exc:  # noqa: BLE001
            log.debug("record_sample skipped | %s: %s", type(exc).__name__, exc)

        symbols = detect_symbols(item.title, self.symbols, item.item.title_ko)
        for sym in symbols:
            self._latest_news[sym] = (item.title, item.score)

        # 강한 감성 뉴스만 진입 평가·우측 로그(임계값 미만·neutral 등은 로그 생략).
        # 소스·키워드 가중치를 반영한 실효 점수로 임계값 비교.
        scored_syms: list[tuple[str, float]] = []
        for sym in symbols:
            eff = effective_news_score(
                item.score,
                source=item.item.source,
                symbol=sym,
                title=title_en,
                title_ko=title_ko,
            )
            if abs(eff) >= settings.news_score_threshold:
                scored_syms.append((sym, eff))
        if not scored_syms:
            return
        if not symbols:
            return
        self._emit_log(
            "INFO", "entry",
            f"진입 평가 · [{item.label}] {self._format_bilingual(title_en, title_ko, item.score)}",
        )
        if not settings.use_news_entry:
            return
        if not self._news_entry_allowed():
            return
        if not self._is_fresh_entry_news(item):
            pub_kst = format_kst(pub, "%H:%M:%S")
            self._emit_log(
                "INFO", "entry",
                f"진입 스킵(구기사): 발행 {pub_kst} · "
                f"{self._format_bilingual(title_en, title_ko, item.score)}",
            )
            return
        for sym, eff_score in scored_syms:
            if sym in self.positions:
                await self._maybe_add(sym, title_en, eff_score, title_ko)
            else:
                await self._maybe_enter(
                    sym, title_en, eff_score, triggered_at, news_ko=title_ko,
                )

    async def _maybe_enter(
        self,
        symbol: str,
        news: str,
        score: float,
        news_triggered_at: datetime | None = None,
        *,
        news_ko: str = "",
    ) -> None:
        if not await self._can_open_slot(symbol):
            return

        ind = await self._indicators(symbol)
        if ind is None:
            return

        # ---- 방향/레버리지 결정 ----
        if settings.auto_leverage:
            # 자동 레버리지: 뉴스 점수 강도로 레버리지를 정하고,
            # |점수|=1.0 이면 역방향이라도 2배로 진입(추세 필터 무시).
            side, leverage = auto_leverage_decision(score, ind.slope)
            if side is None:
                ctx = self._news_context(news, score, news_ko)
                self._emit_log(
                    "INFO", "entry",
                    f"진입 스킵 {symbol}: {ctx} · 자동레버리지 기준 미충족 "
                    f"(기울기 {ind.slope:+.4f})",
                )
                return
        else:
            # 수동 레버리지: 뉴스 방향과 기울기 방향이 일치할 때만 진입.
            side = None
            if score > 0 and ind.slope > 0:
                side = "long"
            elif score < 0 and ind.slope < 0:
                side = "short"
            if side is None:
                ctx = self._news_context(news, score, news_ko)
                self._emit_log(
                    "INFO", "entry",
                    f"진입 스킵 {symbol}: {ctx} · 기울기({ind.slope:+.4f}) 방향 불일치",
                )
                return
            leverage = settings.leverage

        ok, size_mult, reason = await self._entry_size_mult(symbol, side, label="진입")
        if not ok:
            return

        category = classify_news(news, news_ko)
        await self._open(
            symbol, side, ind, news, score, leverage, news_triggered_at,
            news_ko=news_ko, size_mult=size_mult,
            use_hybrid=True, entry_category=category,
        )

    @staticmethod
    def _bb_label(side: Side) -> str:
        return f"BB BREAKOUT {side.upper()}"

    async def _can_open_slot(self, symbol: str) -> bool:
        """동시 포지션 한도·중복 심볼 검사."""
        async with self._lock:
            if symbol in self.positions:
                return False
            if len(self.positions) >= settings.max_positions:
                self._emit_log(
                    "WARNING", "entry",
                    f"진입 보류 {symbol}: 동시 포지션 한도({settings.max_positions}) 도달",
                )
                return False
        return True

    @staticmethod
    def _price_favorable(side: Side, last_price: float, entry_price: float) -> bool:
        if side == "long":
            return last_price > entry_price
        return last_price < entry_price

    async def _entry_size_mult(
        self, symbol: str, side: Side, *, label: str,
    ) -> tuple[bool, float, str]:
        """MTF × 펀딩 게이트. 실패 시 로그 후 (False, 0, reason)."""
        ok, mtf_mult, mtf_reason = await self._mtf_gate(symbol, side)
        if not ok:
            self._emit_log("INFO", "entry", f"{label} 스킵 {symbol}: {mtf_reason}")
            return False, 0.0, mtf_reason
        ok, deriv_mult, deriv_reason = await self._deriv_gate(symbol, side)
        if not ok:
            self._emit_log("INFO", "entry", f"{label} 스킵 {symbol}: {deriv_reason}")
            return False, 0.0, deriv_reason
        return True, mtf_mult * deriv_mult, "ok"

    def _persist_trade(self, trade: dict) -> None:
        """GUI 통계 + JSONL 영속 로그를 한 번에 기록."""
        self.state.record_trade(trade)
        record_trade(trade)

    async def _refresh_live_balance(self) -> None:
        if self.sim or self.engine is None:
            return
        bal, _ = await self.engine.fetch_balance_usdt()
        self.state.set_balance(bal)
        self._last_balance_fetch = asyncio.get_running_loop().time()

    def _apply_exit_price(
        self, pos: Position, exit_price: float, close_notional: float,
    ) -> tuple[float, float, float]:
        """청산가 반영 후 (exit_price, pnl_pct, pnl_usdt)."""
        pos.mark_price = exit_price
        pnl_pct = pos.unrealized_pct()
        return exit_price, pnl_pct, close_notional * (pnl_pct / 100)

    def _log_bb_entry_fail(self, symbol: str, result) -> None:
        """BB 돌파가 감지된 뒤 진입 조건 미충족 시에만 로그를 남긴다."""
        if result.ok or not result.side or not result.reason:
            return
        detail = BB_FAIL_LABELS.get(result.reason, result.reason)
        extra = ""
        if result.reason == "BB_WIDTH":
            extra = f" (bb_min={settings.bb_min}%)"
        elif result.reason == "VOLUME" and result.volume_ratio is not None:
            extra = f" · ratio {result.volume_ratio:.2f} < {settings.vol_mult}"
        elif result.reason == "TREND":
            extra = (
                f" · 모드={settings.bb_trend_mode}, len={settings.f_trend_len}, "
                f"pct={settings.f_trend_pct}"
            )
        elif result.reason == "RANGE":
            extra = f" · min_range={settings.min_range_pct}%"
        self._emit_log(
            "INFO", "entry",
            f"BB 진입 실패 {symbol}: {self._bb_label(result.side)} · {detail}{extra}",
        )

    def _account_balances(self) -> tuple[float, float]:
        """(가용 잔고, 총 평가 잔고) — 오픈 포지션 증거금·미실현 포함."""
        free = self.state.get_balance()
        used = 0.0
        unrealized = 0.0
        for pv in self.state.get_positions():
            lev = max(int(getattr(pv, "leverage", 1) or 1), 1)
            notional = float(getattr(pv, "notional", 0) or 0)
            if notional > 0:
                used += notional / lev
                unrealized += notional * float(getattr(pv, "unrealized_pct", 0) or 0) / 100
        return free, free + used + unrealized

    async def _fetch_ohlcv_1m(self, symbol: str, limit: int = BB_OHLCV_LIMIT) -> list[list[float]] | None:
        if self.sim:
            rows = self.sim_market.ohlcv(symbol, limit=limit)
            return rows if len(rows) >= limit else None
        if self.engine is None:
            return None
        df = await self.engine.fetch_ohlcv_df(symbol, limit=limit, timeframe="1m")
        if df is None or len(df) < limit:
            return None
        return df.values.tolist()

    async def _evaluate_bb_symbol(self, symbol: str, ind) -> None:
        """1m BB 돌파 평가 — 최초 진입 또는 추가 진입."""
        ohlcv = await self._fetch_ohlcv_1m(symbol)
        if not ohlcv:
            return
        bar_ts = int(ohlcv[-1][0])
        if self._bb_last_bar_ts.get(symbol) == bar_ts:
            return
        self._bb_last_bar_ts[symbol] = bar_ts

        result = evaluate_bb_entry(ohlcv)
        pos = self.positions.get(symbol)
        if pos is None:
            if not result.ok:
                self._log_bb_entry_fail(symbol, result)
                return
            await self._maybe_enter_bb(symbol, result.side, ind)
        elif not pos.added:
            if not result.ok:
                if result.side == pos.side:
                    self._log_bb_entry_fail(symbol, result)
                return
            if result.side != pos.side:
                self._emit_log(
                    "INFO", "entry",
                    f"BB 추가진입 실패 {symbol}: 방향 불일치 "
                    f"(보유 {pos.side.upper()} / 신호 {result.side.upper()})",
                )
                return
            await self._maybe_add_bb(symbol, pos, result.side, ind)

    async def _maybe_enter_bb(self, symbol: str, side: Side, ind) -> None:
        if not await self._can_open_slot(symbol):
            return

        ok, size_mult, _ = await self._entry_size_mult(symbol, side, label="BB 진입")
        if not ok:
            return
        label = self._bb_label(side)
        leverage = max(1, int(settings.bb_leverage))
        await self._open(
            symbol, side, ind, label, 0.0, leverage, _now(), news_ko="",
            size_mult=size_mult, entry_category="bb_breakout",
        )

    async def _maybe_add_bb(
        self, symbol: str, pos: Position, side: Side, ind,
    ) -> None:
        if pos.added or side != pos.side:
            return
        label = self._bb_label(side)
        if not self._price_favorable(pos.side, ind.last_price, pos.entry_price):
            self._emit_log(
                "INFO", "entry",
                f"추가진입 보류 {symbol}: 가격 미유리(현재 {ind.last_price:.4f} / "
                f"평균 {pos.entry_price:.4f}) | {label}",
            )
            return
        ok, size_mult, _ = await self._entry_size_mult(symbol, side, label="BB 추가진입")
        if not ok:
            return
        add_lev = min(pos.leverage + 1, settings.bb_max_add_leverage)
        await self._add(pos, ind, label, 0.0, max(1, int(add_lev)), size_mult=size_mult)

    async def _mtf_gate(self, symbol: str, side: Side) -> tuple[bool, float, str]:
        """MTF EMA 필터. (허용, 사이즈배수, 사유)."""
        if not settings.mtf_filter_enabled:
            return True, 1.0, "mtf off"
        now = asyncio.get_running_loop().time()
        cached = self._mtf_cache.get(symbol)
        if cached and now - cached[0] < 60.0:
            trend = cached[1]
        else:
            tfs = parse_mtf_tfs(settings.mtf_tfs)
            if self.sim and self.sim_market is not None:
                # SIM: 1m 합성 종가를 상위 TF로 간주해 동일 시리즈 사용
                rows = self.sim_market.ohlcv(symbol, limit=max(settings.mtf_ema_len + 20, 80))
                df = pd.DataFrame(
                    rows, columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                frames = {tf: df for tf in tfs}
            elif self.engine is not None:
                frames = await self.engine.fetch_mtf_frames(
                    symbol, tfs, ema_len=settings.mtf_ema_len,
                )
            else:
                return True, 1.0, "mtf no engine"
            trend = evaluate_mtf_from_frames(
                frames, ema_len=settings.mtf_ema_len, fallback_len=50,
            )
            self._mtf_cache[symbol] = (now, trend)
        return gate_entry(side, trend, settings)

    async def _deriv_gate(self, symbol: str, side: Side) -> tuple[bool, float, str]:
        """펀딩·OI 필터. (허용, 사이즈배수, 사유)."""
        if not settings.deriv_filter_enabled:
            return True, 1.0, "deriv off"
        if self.sim:
            return True, 1.0, "deriv sim skip"
        if self.engine is None:
            return True, 1.0, "deriv no engine"
        now = asyncio.get_running_loop().time()
        cached = self._deriv_cache.get(symbol)
        if cached and now - cached[0] < max(30, int(settings.oi_cache_sec)):
            snap = cached[1]
        else:
            snap = await self.engine.fetch_derivatives_snapshot(symbol)
            self._deriv_cache[symbol] = (now, snap)
        return deriv_gate_entry(side, snap, settings)

    async def _spike_range(self, symbol: str, ind) -> tuple[float, float]:
        """하이브리드 눌림목용 최근 스파이크 고저."""
        rows = await self._fetch_ohlcv_1m(symbol, limit=8)
        if rows:
            highs = [float(r[2]) for r in rows[-5:]]
            lows = [float(r[3]) for r in rows[-5:]]
            return max(highs), min(lows)
        p = float(ind.last_price)
        return p, p

    # ---- 진입 ----
    async def _open(
        self,
        symbol: str,
        side: Side,
        ind,
        news: str,
        score: float,
        leverage: int,
        news_triggered_at: datetime | None = None,
        *,
        news_ko: str = "",
        size_mult: float = 1.0,
        use_hybrid: bool = False,
        entry_category: str = "default",
    ) -> None:
        atr = ind.atr if ind.atr and not np.isnan(ind.atr) else ind.last_price * 0.01
        notional = compute_notional(
            settings.position_size_usdt,
            atr=atr,
            price=ind.last_price,
            sizing_state=self._sizing,
            extra_mult=size_mult,
        )
        leverage = max(1, int(leverage))
        triggered = news_triggered_at or _now()
        price = ind.last_price

        cat = entry_category
        if cat == "default" and news.startswith("BB BREAKOUT"):
            cat = "bb_breakout"
        overrides = category_exit_overrides(cat)

        hybrid_on = bool(use_hybrid and settings.hybrid_entry_enabled)
        ioc_frac = float(settings.hybrid_ioc_fraction) if hybrid_on else 1.0
        ioc_notional = notional * ioc_frac
        maker_notional = notional - ioc_notional

        order_price = price
        filled = 0.0
        ioc_filled_notional = ioc_notional

        if self.sim:
            if ioc_notional <= 0:
                return
            filled = ioc_notional / price
            margin = ioc_notional / leverage
            self.sim_market.balance -= margin
            self.state.set_balance(self.sim_market.balance)
            order_price = price
        else:
            self._entry_inflight.add(symbol)
            try:
                result = await self.engine.enter_position(
                    symbol, side, leverage=leverage, notional=ioc_notional,
                )
            finally:
                self._entry_inflight.discard(symbol)
            if not result.is_filled:
                ctx = self._news_context(news, score, news_ko)
                self._emit_log(
                    "ERROR", "order",
                    f"진입 실패 {symbol} {side}: {result.reason} | {ctx}",
                )
                return
            order_price = result.price
            filled = result.filled_amount
            ioc_filled_notional = filled * order_price if order_price else ioc_notional
            # 거래소에 실제 포지션이 생겼는지 확인 (고아정리 레이스·허위체결 방지)
            ex_side, ex_amt = await self.engine.get_position_contracts(symbol)
            if ex_side is None or ex_amt <= 0:
                self._emit_log(
                    "ERROR", "order",
                    f"진입 응답은 체결이나 거래소 포지션 없음 {symbol} {side} "
                    f"(filled={filled:.6f} @ {order_price:.4f}) — 내부 등록 취소",
                )
                self.engine.register_exit(symbol)
                return
            if ex_side != side:
                self._emit_log(
                    "ERROR", "order",
                    f"진입 방향 불일치 {symbol}: 주문={side} 거래소={ex_side} — 내부 등록 취소",
                )
                self.engine.register_exit(symbol)
                return
            # 실제 체결 수량으로 보정
            if ex_amt > 0 and abs(ex_amt - filled) / max(ex_amt, filled, 1e-12) > 0.05:
                self._emit_log(
                    "WARNING", "order",
                    f"진입 수량 보정 {symbol}: bot={filled:.8f} → exchange={ex_amt:.8f}",
                )
                filled = ex_amt
                ioc_filled_notional = filled * order_price if order_price else ioc_notional

        async with self._lock:
            pos = Position(
                symbol=symbol, side=side, amount=filled, entry_price=order_price,
                atr=atr,
                entry_news=news, entry_news_ko=news_ko, entry_score=score,
                entry_category=cat,
                news_triggered_at=triggered,
                notional=ioc_filled_notional, leverage=leverage,
                margin=ioc_filled_notional / leverage,
                **overrides,
            )
            pos.prev_rsi = ind.rsi
            self.positions[symbol] = pos
        self.state.clear_closed_chart(symbol)

        ctx = self._news_context(news, score, news_ko) if score or news_ko else news
        hybrid_note = ""
        if hybrid_on and maker_notional > 0:
            spike_high, spike_low = await self._spike_range(symbol, ind)
            pending = PendingEntry.from_spike(
                symbol=symbol, side=side, leverage=leverage,
                maker_notional=maker_notional,
                spike_high=spike_high, spike_low=spike_low,
                news=news, news_ko=news_ko, score=score, category=cat, atr=atr,
            )
            if not self.sim and self.engine is not None:
                maker = await self.engine.submit_maker_limit(
                    symbol, side, leverage=leverage,
                    notional=maker_notional, limit_price=pending.limit_price,
                )
                pending.order_id = maker.order_id
            self._pending_entries[symbol] = pending
            hybrid_note = (
                f" | hybrid IOC {ioc_frac * 100:.0f}% + maker @ {pending.limit_price:.4f}"
            )
        cat_note = f" | cat={cat}" if settings.news_category_tp_enabled else ""
        self._emit_log(
            "INFO", "entry",
            f"진입 {side.upper()} {symbol} | 진입가={order_price:.4f} 수량={filled:.6f} "
            f"금액={ioc_filled_notional:.2f}USDT 레버리지={leverage}x{hybrid_note}{cat_note} | {ctx}",
        )
        self._sync_position_view(self.positions[symbol])

        if not self.sim and self.engine is not None:
            await self._refresh_live_balance()

        if self.notifier is not None:
            await self.notifier.send_position_open(
                symbol=symbol, side=side, amount_usdt=ioc_filled_notional,
                entry_price=order_price, news=news, score=score,
                news_ko=news_ko, leverage=leverage,
            )

    # ---- 추가 진입(피라미딩) ----
    async def _maybe_add(
        self, symbol: str, news: str, score: float, news_ko: str = "",
    ) -> None:
        """보유 중인 포지션에 같은 방향·더 강한 뉴스가 오면 1회 추가 진입한다.

        조건: ① 아직 추가 진입한 적 없음 ② 뉴스 방향이 보유 방향과 동일
        ③ 새 점수 강도가 진입 점수보다 큼 ④ 가격이 진입 방향으로 유리하게 이동.
        충족 시 동일 명목금액을 추가하고 평균단가/손익절 라인을 재계산한다.
        """
        pos = self.positions.get(symbol)
        if pos is None or pos.added:
            return
        # 방향 일치 여부(같은 방향 강세/약세 뉴스인지).
        if (pos.side == "long" and score <= 0) or (pos.side == "short" and score >= 0):
            return
        # 더 강한 확신(절대 점수가 진입 시보다 큼)인지.
        if abs(score) <= abs(pos.entry_score):
            return

        ind = await self._indicators(symbol)
        if ind is None:
            return
        if not self._price_favorable(pos.side, ind.last_price, pos.entry_price):
            ctx = self._news_context(news, score, news_ko)
            self._emit_log(
                "INFO", "entry",
                f"추가진입 보류 {symbol}: 가격 미유리(현재 {ind.last_price:.4f} / "
                f"평균 {pos.entry_price:.4f}) | {ctx}",
            )
            return

        # 추가 분 레버리지: 자동 모드는 점수 강도 기준, 수동 모드는 +1배.
        if settings.auto_leverage:
            add_lev = score_to_leverage(score)
        else:
            add_lev = min(pos.leverage + 1, 25)
        ok, size_mult, _ = await self._entry_size_mult(symbol, pos.side, label="추가진입")
        if not ok:
            return
        await self._add(
            pos, ind, news, score, max(1, int(add_lev)),
            news_ko=news_ko, size_mult=size_mult,
        )

    async def _add(
        self,
        pos: Position,
        ind,
        news: str,
        score: float,
        leverage: int,
        *,
        news_ko: str = "",
        size_mult: float = 1.0,
    ) -> None:
        atr = ind.atr if ind.atr and not np.isnan(ind.atr) else pos.atr
        notional = compute_notional(
            settings.position_size_usdt,
            atr=atr,
            price=ind.last_price,
            sizing_state=self._sizing,
            extra_mult=size_mult,
        )
        leverage = max(1, int(leverage))
        price = ind.last_price
        add_margin = notional / leverage
        if self.sim:
            filled = notional / price
            add_price = price
            self.sim_market.balance -= add_margin
            self.state.set_balance(self.sim_market.balance)
        else:
            result = await self.engine.increase_position(
                pos.symbol, pos.side, leverage=leverage, notional=notional
            )
            if not result.is_filled:
                ctx = self._news_context(news, score, news_ko)
                self._emit_log(
                    "ERROR", "order",
                    f"추가진입 실패 {pos.symbol} {pos.side}: {result.reason} | {ctx}",
                )
                return
            add_price = result.price
            filled = result.filled_amount

        async with self._lock:
            prev_entry = pos.entry_price
            pos.add_fill(
                add_amount=filled, add_price=add_price, add_notional=notional,
                add_margin=add_margin, leverage=leverage, atr=atr,
            )
            pos.prev_rsi = ind.rsi
            if news_ko:
                pos.entry_news_ko = news_ko

        ctx = self._news_context(news, score, news_ko) if score or news_ko else news
        self._emit_log(
            "INFO", "entry",
            f"➕ 추가진입 {pos.side.upper()} {pos.symbol} | 추가가={add_price:.4f} "
            f"수량+={filled:.6f} 평균가={prev_entry:.4f}→{pos.entry_price:.4f} "
            f"총금액={pos.notional:.2f}USDT 배율={leverage}x | {ctx}",
        )
        self._sync_position_view(pos)

        if self.notifier is not None:
            await self.notifier.send_position_open(
                symbol=pos.symbol, side=pos.side, amount_usdt=notional,
                entry_price=add_price, news=news, score=score,
                news_ko=news_ko or pos.entry_news_ko, leverage=leverage,
            )

    # ---- 모니터 루프(가격 갱신 + 청산 판정) ----
    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                await self._monitor_once()
            except Exception as exc:  # noqa: BLE001 - 루프 생존
                log_exception(log, exc, context="monitor_loop")
            await asyncio.sleep(max(1, int(settings.monitor_interval)))

    async def _monitor_once(self) -> None:
        # ---- 수동 청산 요청 처리(GUI 버튼) ----
        for req_sym in self.state.pop_close_requests():
            pos = self.positions.get(req_sym)
            if pos is not None:
                await self._close(
                    pos,
                    ExitSignal(
                        True, reason="GUI 수동 청산 요청",
                        exit_type="manual", order_type="market",
                    ),
                )

        for symbol in self.symbols:
            # 가격 진행(SIM) 및 최신 지표 계산.
            if self.sim:
                self.sim_market.tick(symbol)
            ind = await self._indicators(symbol)
            if ind is None:
                continue

            ohlcv = self._ohlcv_cache.get(symbol, [])
            if ohlcv:
                self.state.set_ohlcv(symbol, ohlcv)

            if settings.use_bb_entry:
                await self._evaluate_bb_symbol(symbol, ind)

            await self._monitor_pending_entry(symbol, ind)

            pos = self.positions.get(symbol)
            if pos is None:
                continue

            news_title, news_score = self._latest_news[symbol]
            signal = pos.update(
                ind.last_price, atr=ind.atr, slope=ind.slope, rsi=ind.rsi,
                news_score=news_score,
            )
            self._sync_position_view(pos)

            if signal.should_exit:
                await self._close(pos, signal)

        # 잔고 갱신(LIVE) — 60초마다 1회(모니터 주기와 분리).
        if not self.sim and self.engine is not None:
            loop = asyncio.get_running_loop()
            if loop.time() - self._last_balance_fetch >= self._balance_poll_sec:
                self._last_balance_fetch = loop.time()
                bal, bal_err = await self.engine.fetch_balance_usdt()
                self.state.set_balance(bal)
                if bal_err:
                    self._emit_log("ERROR", "system", f"잔고 조회 실패: {bal_err}")
            if loop.time() - self._last_orphan_check >= self._orphan_poll_sec:
                self._last_orphan_check = loop.time()
                await self._reconcile_orphan_positions()

    async def _reconcile_orphan_positions(self) -> None:
        """봇이 추적하지 않는 거래소 잔여(dust) 포지션을 시장가로 정리한다.

        진입 직후(``_entry_inflight`` / 엔진 슬롯 / pending Maker)는 절대 건드리지
        않는다 — 이전 버전은 이 레이스로 방금 체결된 포지션을 바로 청산했다.
        """
        if self.sim or self.engine is None:
            return
        try:
            open_pos = await self.engine.fetch_open_positions()
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="orphan_reconcile_fetch")
            return

        tracked_n = {normalize_symbol_key(s) for s in self.positions}
        tracked_n |= {normalize_symbol_key(s) for s in self._entry_inflight}
        tracked_n |= {normalize_symbol_key(s) for s in self._pending_entries}
        tracked_n |= {normalize_symbol_key(s) for s in self.engine.tracked_symbols()}
        for p in open_pos:
            raw_sym = str(p.get("symbol") or "")
            key = normalize_symbol_key(raw_sym)
            if key in tracked_n:
                continue
            raw_side = (p.get("side") or "").lower()
            contracts = abs(float(p.get("contracts") or 0))
            if contracts <= 0 or raw_side not in ("long", "short"):
                continue
            bot_sym = next(
                (s for s in self.symbols if normalize_symbol_key(s) == key),
                raw_sym.split(":")[0] if ":" in raw_sym else raw_sym,
            )
            self._emit_log(
                "WARNING", "system",
                f"고아 포지션 감지 → 시장가 정리 {bot_sym} {raw_side} x{contracts:.8f}",
            )
            result = await self.engine.close_position(
                bot_sym, raw_side, contracts,  # type: ignore[arg-type]
                order_type="market", register_exit=True, verify_flat=True,
            )
            if result.is_filled:
                self._emit_log(
                    "WARNING", "exit",
                    f"고아 포지션 청산 완료 {bot_sym} @ {result.price:.4f}",
                )
            else:
                self._emit_log(
                    "ERROR", "order",
                    f"고아 포지션 청산 실패 {bot_sym}: {result.reason}",
                )

    async def _monitor_pending_entry(self, symbol: str, ind) -> None:
        """하이브리드 Maker 2차 체결 대기·만료 처리."""
        pending = self._pending_entries.get(symbol)
        if pending is None:
            return
        now = _now()
        if pending.expired(now):
            if not self.sim and self.engine is not None:
                await self.engine.cancel_order_safe(symbol, pending.order_id)
            self._pending_entries.pop(symbol, None)
            self._emit_log(
                "INFO", "entry",
                f"하이브리드 Maker 만료 {symbol} @ {pending.limit_price:.4f}",
            )
            return

        pos = self.positions.get(symbol)
        if pos is None:
            return

        filled = False
        fill_price = pending.limit_price
        fill_amount = 0.0

        if not self.sim and self.engine is not None and pending.order_id:
            result = await self.engine.fetch_order_fill(symbol, pending.order_id)
            if result and result.is_filled:
                filled = True
                fill_price = result.price or pending.limit_price
                fill_amount = result.filled_amount
        else:
            mark = ind.last_price
            touched = (
                (pending.side == "long" and mark <= pending.limit_price)
                or (pending.side == "short" and mark >= pending.limit_price)
            )
            if touched:
                filled = True
                fill_price = pending.limit_price
                fill_amount = pending.maker_notional / fill_price

        if not filled or fill_amount <= 0:
            return

        add_margin = pending.maker_notional / max(pending.leverage, 1)
        if self.sim:
            self.sim_market.balance -= add_margin
            self.state.set_balance(self.sim_market.balance)

        async with self._lock:
            pos.add_fill(
                add_amount=fill_amount, add_price=fill_price,
                add_notional=pending.maker_notional, add_margin=add_margin,
                leverage=pending.leverage, atr=pending.atr or pos.atr,
            )
        if not self.sim and self.engine is not None and pending.order_id:
            await self.engine.cancel_order_safe(symbol, pending.order_id)
        self._pending_entries.pop(symbol, None)
        self._sync_position_view(pos)
        self._emit_log(
            "INFO", "entry",
            f"하이브리드 Maker 체결 {pos.side.upper()} {symbol} | "
            f"가={fill_price:.4f} +{fill_amount:.6f} 총={pos.notional:.2f}USDT",
        )

    # ---- 청산 ----
    async def _close(self, pos: Position, signal) -> None:
        frac = getattr(signal, "close_fraction", None)
        is_partial = (
            signal.exit_type == "scale_out"
            and frac is not None
            and 0.05 <= float(frac) < 1.0
            and not pos.scale_out_done
        )
        close_amount = pos.amount * float(frac) if is_partial else pos.amount
        close_notional = pos.notional * float(frac) if is_partial else pos.notional
        close_margin = (
            (pos.margin if pos.margin > 0 else pos.notional / max(pos.leverage, 1))
            * (float(frac) if is_partial else 1.0)
        )

        exit_price = pos.mark_price
        pnl_pct = pos.unrealized_pct()
        pnl_usdt = close_notional * (pnl_pct / 100)

        if self.sim:
            self.sim_market.balance += close_margin + pnl_usdt
            self.state.set_balance(self.sim_market.balance)
        else:
            result = await self.engine.close_position(
                pos.symbol, pos.side, close_amount,
                order_type=signal.order_type,
                register_exit=not is_partial,
                verify_flat=not is_partial,
            )
            if not result.is_filled:
                self._emit_log(
                    "ERROR", "order",
                    f"청산 실패 {pos.symbol}: {result.reason}",
                )
                return
            if result.price:
                exit_price, pnl_pct, pnl_usdt = self._apply_exit_price(
                    pos, result.price, close_notional,
                )

        if is_partial:
            async with self._lock:
                pos.apply_scale_out(float(frac))
            self._sizing.stats.record(pnl_pct)
            self._sync_position_view(pos)
            self._emit_log(
                "INFO", "exit",
                f"부분익절 {pos.side.upper()} {pos.symbol} | "
                f"{float(frac) * 100:.0f}% @ {exit_price:.4f} "
                f"PnL={pnl_pct:+.2f}% ({pnl_usdt:+.2f}USDT) | 잔량={pos.amount:.6f} | "
                f"{signal.reason}",
            )
            self._persist_trade({
                "symbol": pos.symbol,
                "side": pos.side,
                "leverage": pos.leverage,
                "notional": close_notional,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "pnl_usdt": pnl_usdt,
                "exit_type": "scale_out",
                "partial": True,
                "entry_category": pos.entry_category,
            })
            await self._refresh_live_balance()
            return

        pending = self._pending_entries.pop(pos.symbol, None)
        if pending is not None and not self.sim and self.engine is not None:
            await self.engine.cancel_order_safe(pos.symbol, pending.order_id)

        async with self._lock:
            self.positions.pop(pos.symbol, None)
        self.state.set_closed_chart(
            ClosedChartView(
                symbol=pos.symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                stop_loss=pos.stop_loss_price,
                trailing_stop=pos.trailing_stop,
                trailing_active=pos.trailing_active,
                entry_news=pos.entry_news,
                entry_news_ko=pos.entry_news_ko,
                entry_score=pos.entry_score,
                opened_at=format_kst(pos.opened_at),
                opened_at_ms=int(pos.opened_at.timestamp() * 1000),
                closed_at_ms=int(_now().timestamp() * 1000),
                news_triggered_at_ms=int(pos.news_triggered_at.timestamp() * 1000),
                exit_type=signal.exit_type,
                pnl_pct=pnl_pct,
            )
        )
        self.state.remove_position(pos.symbol)
        self._sizing.stats.record(pnl_pct)
        await self._refresh_live_balance()

        self._persist_trade({
            "symbol": pos.symbol,
            "side": pos.side,
            "leverage": pos.leverage,
            "notional": pos.notional,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "pnl_usdt": pnl_usdt,
            "exit_type": signal.exit_type,
            "reason": signal.reason,
            "opened_at": pos.opened_at.isoformat(),
            "closed_at": _now().isoformat(),
            "added": pos.added,
            "entry_category": pos.entry_category,
            "entry_news": pos.entry_news,
        })

        ctx = self._news_context(pos.entry_news, pos.entry_score, pos.entry_news_ko)
        self._emit_log(
            "INFO", "exit",
            f"청산 {pos.side.upper()} {pos.symbol} | 진입가={pos.entry_price:.4f} "
            f"청산가={exit_price:.4f} 손익={pnl_pct:+.2f}% ({pnl_usdt:+.2f}USDT) "
            f"사유={signal.exit_type} ({signal.reason}) | {ctx}",
        )

        if self.notifier is not None:
            bal_free, bal_equity = self._account_balances()
            await self.notifier.send_position_close(
                symbol=pos.symbol, side=pos.side, amount_usdt=pos.notional,
                entry_price=pos.entry_price, exit_price=exit_price, pnl_pct=pnl_pct,
                reason=f"{signal.exit_type}: {signal.reason}",
                news=pos.entry_news, score=pos.entry_score,
                pnl_usdt=pnl_usdt, news_ko=pos.entry_news_ko,
                balance_free=bal_free, balance_equity=bal_equity,
            )

    # ---- 헬퍼 ----
    async def _indicators(self, symbol: str):
        if self.sim:
            ohlcv = self.sim_market.ohlcv(symbol)
            self._ohlcv_cache[symbol] = ohlcv
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            return compute_indicators_from_df(symbol, settings.timeframe, df)
        df = await self.engine.fetch_ohlcv_df(symbol)
        if df is None:
            return None
        self._ohlcv_cache[symbol] = df.values.tolist()
        if len(df) < max(RSI_LENGTH, ATR_LENGTH) + 1:
            return None
        return compute_indicators_from_df(symbol, settings.timeframe, df)

    def _sync_position_view(self, pos: Position) -> None:
        self.state.upsert_position(
            PositionView(
                symbol=pos.symbol, side=pos.side, amount=pos.amount,
                entry_price=pos.entry_price, mark_price=pos.mark_price,
                stop_loss=pos.stop_loss_price, trailing_stop=pos.trailing_stop,
                atr_mult=pos.atr_mult, unrealized_pct=pos.unrealized_pct(),
                entry_news=pos.entry_news, entry_news_ko=pos.entry_news_ko,
                entry_score=pos.entry_score,
                opened_at=format_kst(pos.opened_at),
                leverage=pos.leverage,
                opened_at_ms=int(pos.opened_at.timestamp() * 1000),
                news_triggered_at_ms=int(pos.news_triggered_at.timestamp() * 1000),
                added=pos.added,
                notional=pos.notional,
                trailing_active=pos.trailing_active,
            )
        )

    def _emit_log(self, level: str, category: str, message: str) -> None:
        self.state.log(level, category, message)
        getattr(log, level.lower(), log.info)("[%s] %s", category, message)


async def _main() -> None:
    bot = TradingBot()
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("Stopped by user")
