"""카테고리별 청산 프로필 그리드 탐색 (스켈레톤).

사용 예:
    python scripts/category_exit_tune.py --days 90

실제 백테스트 연동은 trade_log.jsonl 누적 데이터 또는
bb_trend_tune 스타일 fast sim 확장 후 구현한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from news_category import _DEFAULT_PROFILES, category_exit_overrides  # noqa: E402
from trade_log import read_trades  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Category exit profile tune skeleton")
    parser.add_argument("--days", type=int, default=90, help="분석 기간(일)")
    args = parser.parse_args()

    trades = read_trades()
    if not trades:
        print("No trades in logs/trades.jsonl — run live/SIM bot first.")
        print("Default profiles:")
        print(json.dumps(_DEFAULT_PROFILES, indent=2, ensure_ascii=False))
        return

    by_cat: dict[str, list[float]] = {}
    for t in trades[-5000:]:
        cat = t.get("entry_category") or "default"
        pnl = float(t.get("pnl_pct") or 0.0)
        by_cat.setdefault(cat, []).append(pnl)

    print(f"Trades loaded: {len(trades)} (window ~{args.days}d requested)")
    for cat, pnls in sorted(by_cat.items()):
        avg = sum(pnls) / len(pnls) if pnls else 0.0
        print(f"  {cat:12s} n={len(pnls):4d} avg_pnl={avg:+.2f}%")
        prof = category_exit_overrides(cat)
        if prof:
            print(f"    profile keys: {list(prof.keys())}")


if __name__ == "__main__":
    main()
