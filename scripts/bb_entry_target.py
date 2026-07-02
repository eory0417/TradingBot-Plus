"""BB mult 이진 탐색 — 90일·코인당 ~40진입 목표."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bb_entry_tune import (  # noqa: E402
    FIXED,
    SYMBOLS,
    TARGET_ENTRIES,
    SymbolData,
    _base_cfg,
    count_entries_fast,
)

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def main() -> None:
    cache: dict[str, SymbolData] = {}
    for sym in SYMBOLS:
        key = f"{sym.replace('/', '_')}_90d.pkl"
        path = CACHE_DIR / key
        if not path.exists():
            print(f"missing cache {path}")
            return
        with path.open("rb") as f:
            cache[sym] = pickle.load(f)

    base = dict(bb_len=20, bb_min=0.5, vol_mult=1.2, vol_len=15, min_range_pct=0.05)

    print(f"Target: >={TARGET_ENTRIES} entries/coin, prefer closest to ~45\n")
    candidates: list[dict] = []

    for bb_len in [15, 20, 25]:
        for bb_min in [0.5, 0.8, 1.0, 1.2]:
            for vol_mult in [1.2, 1.5, 1.8]:
                lo, hi = 1.2, 3.5
                best_row = None
                for _ in range(12):
                    mid = round((lo + hi) / 2, 2)
                    cfg = _base_cfg()
                    for k, v in {**base, "bb_len": bb_len, "bb_min": bb_min, "vol_mult": vol_mult, "bb_mult": mid}.items():
                        setattr(cfg, k, v)
                    per = {s: count_entries_fast(cache[s], cfg) for s in SYMBOLS}
                    mn = min(per.values())
                    row = {"bb_len": bb_len, "bb_mult": mid, "bb_min": bb_min, "vol_mult": vol_mult, **per, "min": mn}
                    if mn >= TARGET_ENTRIES:
                        best_row = row
                        lo = mid  # tighter bands -> more entries, search higher mult to reduce
                    else:
                        hi = mid
                if best_row:
                    best_row["avg"] = sum(best_row[s] for s in SYMBOLS) / len(SYMBOLS)
                    candidates.append(best_row)

    if not candidates:
        print("No combo found >=40 with tested ranges. Try looser filters.")
        return

    # pick: min>=40, then avg closest to 45
    candidates.sort(key=lambda r: (abs(r["avg"] - 45), abs(r["min"] - 40)))
    best = candidates[0]
    print("RECOMMENDED:")
    for k in ["bb_len", "bb_mult", "bb_min", "vol_mult", "vol_len", "min_range_pct"]:
        print(f"  {k}: {best.get(k, base.get(k, FIXED.get(k)))}")
    print("  per_symbol:", {s: best[s] for s in SYMBOLS})
    print(f"  min={best['min']} avg={best['avg']:.1f}")


if __name__ == "__main__":
    main()
