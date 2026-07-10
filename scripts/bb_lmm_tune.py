"""BB Len / Mult / Min 그리드 탐색 (고속) — 20/3.2/1.1 주변 촘촘히.

나머지 설정은 현재 추천값 고정. 90일 · 4코인 합산 순손익.
예상: ~495조합 × ~15s ≈ 2시간.
"""

from __future__ import annotations

import itertools
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bb_trend_tune import (  # noqa: E402
    CACHE_DIR,
    DAYS,
    FAST_KEY,
    FIXED,
    SYMBOLS,
    FastSymbol,
    _build_fast,
    _load_frames,
    _run_combo,
)
import bb_trend_tune  # noqa: E402
import pickle  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "bb_lmm_tune_out.txt"

# 20 / 3.2 / 1.1 주변 — 5 × 11 × 9 = 495 combos
BB_LENS = [18, 19, 20, 21, 22]
BB_MULTS = [round(2.7 + i * 0.1, 1) for i in range(11)]  # 2.7 … 3.7
BB_MINS = [round(0.7 + i * 0.1, 1) for i in range(9)]    # 0.7 … 1.5
BASELINE = dict(bb_len=20, bb_mult=3.2, bb_min=1.1)
LOG_EVERY = 25


def _load_fast_cache(since: int, until: int, frames: dict) -> dict[str, FastSymbol]:
    path = CACHE_DIR / FAST_KEY
    if path.exists():
        print(f"fast cache hit {path}", flush=True)

        class _PU(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if module == "__main__":
                    return getattr(bb_trend_tune, name)
                return super().find_class(module, name)

        with path.open("rb") as f:
            return _PU(f).load()
    return _build_fast(frames, since, until)


def _combos() -> list[dict]:
    return [
        dict(bb_len=ln, bb_mult=mult, bb_min=bmin)
        for ln, mult, bmin in itertools.product(BB_LENS, BB_MULTS, BB_MINS)
    ]


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

    log(f"BB L/M/M tune (fast) | {DAYS}d | {SYMBOLS}")
    log(
        f"fixed: trend={FIXED.get('bb_trend_mode', 'relaxed')} "
        f"trail={FIXED['trailing_profit_pct']}% time={FIXED['time_exit_hours']}h "
        f"MTF={FIXED['mtf_mode']} scale={FIXED['scale_out_fraction']}"
    )
    log(f"grid: len={BB_LENS}")
    log(f"      mult={BB_MULTS[0]}…{BB_MULTS[-1]} (step 0.1)")
    log(f"      min={BB_MINS[0]}…{BB_MINS[-1]} (step 0.1)")
    log(f"→ {len(combos)} combos")

    frames = _load_frames(since, until)
    t0 = time.perf_counter()
    cache = _load_fast_cache(since, until, frames)
    log(f"precompute done in {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    base_probe = _run_combo(cache, BASELINE)
    one = time.perf_counter() - t1
    log(f"baseline {BASELINE} → PnL {base_probe['pnl']:+.2f} USDT ({one:.1f}s)")
    log(f"1 combo ≈ {one:.1f}s → 전체 예상 ≈ {one * len(combos) / 3600:.2f} h")

    results: list[dict] = []
    t_all = time.perf_counter()
    for i, ov in enumerate(combos, 1):
        t2 = time.perf_counter()
        row = _run_combo(cache, ov)
        results.append(row)
        if i == 1 or i % LOG_EVERY == 0 or i == len(combos):
            log(
                f"[{i:03d}/{len(combos)}] "
                f"len={row['bb_len']:2d} mult={row['bb_mult']:.1f} min={row['bb_min']:.1f} | "
                f"PnL {row['pnl']:+7.2f} | trades={row['trades']:3d} WR={row['wr']:5.1f}% "
                f"({time.perf_counter() - t2:.1f}s)"
            )

    results.sort(key=lambda r: r["pnl"], reverse=True)
    log("")
    log("=" * 72)
    log("TOP 15 (합산 순손익)")
    log("=" * 72)
    for i, r in enumerate(results[:15], 1):
        log(
            f"{i:2d}. len={r['bb_len']:2d} mult={r['bb_mult']:.1f} min={r['bb_min']:.1f} | "
            f"PnL {r['pnl']:+7.2f} | trades={r['trades']:3d} WR={r['wr']:5.1f}%"
        )
        for sym, p in r["per"].items():
            log(f"      {sym}: {p['pnl']:+.2f} ({p['trades']}t, WR {p['wr']:.1f}%)")

    best = results[0]
    base_row = next(
        r for r in results
        if r["bb_len"] == BASELINE["bb_len"]
        and r["bb_mult"] == BASELINE["bb_mult"]
        and r["bb_min"] == BASELINE["bb_min"]
    )
    log("")
    log(
        f"BEST: len={best['bb_len']} mult={best['bb_mult']} min={best['bb_min']} "
        f"→ {best['pnl']:+.2f} USDT"
    )
    log(
        f"BASELINE (20/3.2/1.1): {base_row['pnl']:+.2f} USDT | "
        f"delta={best['pnl'] - base_row['pnl']:+.2f}"
    )
    log(f"elapsed {time.perf_counter() - t_all:.1f}s ({(time.perf_counter() - t_all) / 3600:.2f} h)")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
