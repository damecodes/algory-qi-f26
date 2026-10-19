# HW4: a walk-forward backtester for S&P 500 timing rules

This folder contains a small backtester for rules that decide each day whether to hold the S&P 500 total-return index (`^SP500TR`) or hold cash. It also contains an honest train/test evaluation of one such rule.

## What it does

- **`load_and_split`** downloads daily closes since 1990. It splits them at 2011-09-09 into a training set and a test set. The test set starts 60 days early, so the strategy's 60-day lookback window is already full on the first out-of-sample day.
- **`backtest(prices, strategy)`** walks forward one day at a time. On each day it hands the strategy the last 60 closes, ending with today's close. If the strategy says hold, it applies tomorrow's return; otherwise the return is zero. The strategy never sees the price it is predicting. It reports:
  - total return
  - annualised return (252 trading days a year, compounded)
  - max drawdown
  - fraction of days invested
- **Control strategies** check that the backtester is right. Always-hold must match the index exactly, always-cash must return exactly 0%, and random-70% must land between the two.
- **17 moving-average trend variants** are all tested on the training set and all printed. The winner was chosen by a rule fixed in advance (highest training annualised return). That winner, "40-day SMA rising over 20 days", was then run once on the test set.

## Result

On the test set (2011–2026) the rule returned **+8.68% a year**, against **+15.47% a year** for buy-and-hold. It lost. The write-up explains why. See [writeup.md](writeup.md).

## Run it

```bash
python hw04_starter.py
```

```bash
python check_hw04.py
```

You need `numpy`, `pandas`, `matplotlib` and `yfinance`. `_check.py` is a minimal stand-in for the course's checker helper.

## Files

| File | What it is |
|------|------------|
| `hw04_starter.py` | the backtester, all strategies, and the Q1–Q9 output |
| `check_hw04.py`, `_check.py` | self-checks (23/23 pass) |
| `hw04_output.txt` | printed output of the run |
| `writeup.md` | answers and the Q10 write-up |
| `q*.png` | plots |
