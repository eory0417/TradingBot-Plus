# Phase1 validation (BTC/USDT, 90d)

Common: BB 20/3.2/1.1 vol1.5, SL 2%, trail 2%/ATR2, time 3h, fee 0.04%, slip 0.02%

| Case | Net USDT | Trades | Win rate | Notes |
|------|----------|--------|----------|-------|
| BASELINE (no MTF, no scale-out, fixed size) | -1.50 | 40 | 50.0% | |
| SCALE-OUT only (40% @ ATR×1.5, move BE) | **+0.94** | 61 | 59.0% | Best on this window |
| MTF block + SCALE-OUT + VOL sizing | -1.40 | 34 | 58.8% | Fewer entries (trend filter) |

Conclusion: Scale-out improved BTC 90d PnL vs baseline. MTF block reduced trade count; combined with vol sizing was near baseline on this sample. Prefer enabling scale-out by default; tune MTF mode (block vs reduce) per regime.
