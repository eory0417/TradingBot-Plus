import itertools
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bb_entry_tune import SYMBOLS, SymbolData, _base_cfg, count_entries_fast

cache = {}
for sym in SYMBOLS:
    fname = sym.replace("/", "_") + "_90d.pkl"
    with open(ROOT / "scripts" / "cache" / fname, "rb") as f:
        cache[sym] = pickle.load(f)

TARGET = 40
IDEAL = 45

rows = []
for mult, bmin, vm, bl in itertools.product(
    [3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
    [1.0, 1.1, 1.2, 1.3, 1.4],
    [1.5, 1.8, 2.0],
    [20],
):
    cfg = _base_cfg()
    for k, v in dict(
        bb_len=bl, bb_mult=mult, bb_min=bmin, vol_mult=vm,
        vol_len=15, min_range_pct=0.05,
    ).items():
        setattr(cfg, k, v)
    per = {s: count_entries_fast(cache[s], cfg) for s in SYMBOLS}
    mn = min(per.values())
    avg = sum(per.values()) / 4
    if mn >= TARGET:
        rows.append((abs(avg - IDEAL), abs(mn - TARGET), mult, bmin, vm, bl, mn, avg, per))

rows.sort()
print(f"hits (>={TARGET}) sorted by closeness to avg~{IDEAL}:\n")
for r in rows[:20]:
    _, _, mult, bmin, vm, bl, mn, avg, per = r
    print(f"mult={mult} bb_min={bmin} vol={vm} len={bl} -> min={mn} avg={avg:.1f} {per}")

if not rows:
    print("no hit - showing closest below target")
    best = None
    for mult, bmin, vm in itertools.product([2.8, 3.0, 3.2, 3.4], [1.0, 1.2, 1.4], [1.5, 2.0]):
        cfg = _base_cfg()
        for k, v in dict(bb_len=20, bb_mult=mult, bb_min=bmin, vol_mult=vm, vol_len=15, min_range_pct=0.05).items():
            setattr(cfg, k, v)
        per = {s: count_entries_fast(cache[s], cfg) for s in SYMBOLS}
        mn = min(per.values())
        if best is None or mn > best[0]:
            best = (mn, mult, bmin, vm, per)
    print(best)
