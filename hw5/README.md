# HW5: reading a real option chain

Payoff functions for the four single-option positions, then a live SPY call chain split into intrinsic and extrinsic value at a one-month and an eight-month expiry.

## Result

Run on 2026-10-25 (Friday 2026-10-23 close), SPY spot $759.97:

- Extrinsic value peaks at the 760 strike, right at spot: **$16.08** for 2026-11-27 (33 days) and **$51.75** for 2027-06-17 (235 days).
- The long-dated curve is higher at all 47 strikes the two chains share.

## Run it

```bash
python hw05_starter.py              # live chain from yfinance, saves data/
python hw05_starter.py --snapshot   # rerun on the saved chains
python check_hw05.py                # 27 offline checks
```

You need `numpy`, `pandas`, `matplotlib` and `yfinance`. The official starter and checker were not available, so both were written from the assignment text. `_check.py` is the same stand-in helper as HW4.

## Data cleaning

- Calls with a zero bid are dropped, as the assignment asks.
- Calls with no trade in the 7 days before the run are also dropped. On a weekend yfinance serves each contract's quote from its last trade, and some deep in-the-money June 2027 strikes last traded in March. Against today's spot they showed a mid up to $48 below intrinsic.

## Files

| File | What it is |
|------|------------|
| `hw05_starter.py` | `payoff`, chain loading, the intrinsic/extrinsic split, Q1–Q6 output |
| `check_hw05.py`, `_check.py` | self-checks (27/27 pass) |
| `hw05_output.txt` | printed output of the live run |
| `data/` | the two saved call chains and the run date |
| `q2_payoffs.png`, `q5_extrinsic_short.png`, `q6_extrinsic_both.png` | plots |
