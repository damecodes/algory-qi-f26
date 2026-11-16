# HW8: price an option from history alone

This folder holds:

- `n_day_returns`, which builds every overlapping N-day return from 20 years of daily SPY closes;
- `price_from_history`, which applies each historical return to today's spot and averages the option payoff, with no distributional assumption;
- a comparison with Black-Scholes from HW6, using 60-day realised volatility, for one contract and across strikes from 15% below spot to 15% above;
- a check of how many independent events actually sit in the tails.

## Result

The run used raw SPY closes from 2006-11-15 to Friday 2026-11-13 (5,030 days, 5,009 overlapping 21-day windows) and the assignment's spot of $760.88.

- **Shape:** 21-day returns have a skew of **−1.19** and an excess kurtosis of **+5.87**. Daily returns have an excess kurtosis of **+14.6**.
- **3%-out-of-the-money 21-day contracts (strike $783.71):** history prices the call at **$5.36** and the put at **$21.92**, undiscounted. Black-Scholes at 14.92% vol gives **$4.86** and **$27.69**. Most of that gap is SPY's +0.82% average monthly drift, which the historical method keeps.
- **Across strikes:** the methods roughly agree near the money. The 15%-out-of-the-money put is **772 times** dearer from history ($0.44 against $0.0006). Read through Black-Scholes, the historical prices trace out the same skew as HW6's live implied-vol curve.
- **Tails:** the 22 windows with a drop of more than 20% come from just **2 episodes** (2008 and 2020), so deep-tail prices rest on very little independent data.

[writeup.md](writeup.md) has the answers, in text only. `HW8_writeup.pdf` is the same write-up as a PDF, and the plots are separate PNG files.

## Run it

Download 20 years of SPY and the latest T-bill rate, save them to `data/`, and run Q1–Q7:

```bash
python hw08_starter.py
```

Rerun on the saved data instead:

```bash
python hw08_starter.py --snapshot
```

Run the 10 self-checks:

```bash
python check_hw08.py
```

You need `numpy`, `pandas`, `scipy`, `matplotlib` and `yfinance`. Q5 and Q6 import `black_scholes()` and `implied_volatility()` from `../hw6/hw06_starter.py`, so keep the two folders side by side. `_check.py` is the same stand-in helper as in HW4 and HW7.

## Files

| File | What it is |
|------|------------|
| `hw08_starter.py` | the two pricing functions, data download, and the Q1–Q7 output and plots |
| `check_hw08.py`, `_check.py` | self-checks (10/10 pass) |
| `hw08_output.txt` | printed output of the run |
| `data/` | saved SPY closes (raw and adjusted) and run metadata, including the T-bill rate |
| `writeup.md`, `HW8_writeup.pdf` | the answers, in text only |
| `q2_return_distribution.png` | 21-day returns against a fitted normal, on linear and log scales |
| `q4_final_prices.png` | final-price distribution for the call and put, with the paying region shaded |
| `q6_strike_comparison.png` | prices, differences and historical implied vol across strikes |
