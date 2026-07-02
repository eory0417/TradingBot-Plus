import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bb_entry_tune import SYMBOLS, SymbolData, _base_cfg, count_entries_fast  # noqa: F401

cache = {}
for sym in SYMBOLS:
    fname = sym.replace("/", "_") + "_90d.pkl"
    with open(ROOT / "scripts" / "cache" / fname, "rb") as f:
        cache[sym] = pickle.load(f)


def test(**p):
    cfg = _base_cfg()
    base = dict(bb_len=20, bb_mult=2.0, bb_min=1.0, vol_mult=1.5, vol_len=15, min_range_pct=0.05)
    base.update(p)
    for k, v in base.items():
        setattr(cfg, k, v)
    per = {s: count_entries_fast(cache[s], cfg) for s in SYMBOLS}
    return per, min(per.values()), sum(per.values()) / 4


print("mult sweep (bb_len=20 bb_min=1.0 vol=1.5)")
for mult in [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5]:
    per, mn, avg = test(bb_mult=mult)
    print(f"  mult={mult}: min={mn} avg={avg:.0f}  {per}")

print("\nbb_min sweep (mult=2.2)")
for bmin in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]:
    per, mn, avg = test(bb_mult=2.2, bb_min=bmin)
    print(f"  bb_min={bmin}: min={mn} avg={avg:.0f}  {per}")

print("\nvol_mult sweep (mult=2.2 bb_min=1.0)")
for vm in [1.0, 1.2, 1.5, 1.8, 2.0]:
    per, mn, avg = test(bb_mult=2.2, bb_min=1.0, vol_mult=vm)
    print(f"  vol={vm}: min={mn} avg={avg:.0f}  {per}")

print("\nbb_len sweep (mult=2.2 bb_min=1.0 vol=1.5)")
for bl in [15, 20, 25, 30]:
    per, mn, avg = test(bb_len=bl, bb_mult=2.2, bb_min=1.0)
    print(f"  len={bl}: min={mn} avg={avg:.0f}  {per}")
