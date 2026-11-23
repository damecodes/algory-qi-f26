# HW9: volatility, tails, and value at risk

This folder holds:

- the five functions the checker tests: `recovery_needed`, `annualise_vol`, `var`, `cvar` and `simulate_paths` (GBM);
- `bootstrap_paths`, which resamples actual daily returns, one day at a time or in blocks;
- Q1–Q9 on 20 years of daily SPY total returns, with four plots and a written answer to Q10.

`simulate_paths`, `bootstrap_paths`, `var` and `cvar` are meant to be reused for position sizing in Session 10.

## Result

The run used dividend-adjusted SPY closes from 2006-11-24 to Friday 2026-11-20 (5,028 daily returns).

- **Volatility:** daily volatility is **1.23%**, or **19.52%** a year when scaled by √252. Measured directly from 19 calendar years it is **17.73%**, a ratio of **0.91**. That is the direction Homework 3's mean reversion predicts, but the confidence interval is too wide to call the gap significant.
- **Clustering:** 60-day volatility peaked at **76%** (2008-12-08, financial crisis) and **61%** (2020-05-18, COVID crash), and bottomed at **5%** (2017-11-15).
- **Tails:** excess kurtosis is **+14.6**. There were **38** days beyond four standard deviations, where a normal distribution predicts **0.32**, so SPY had **119 times** as many.
- **Compounding:** a 50% loss needs a 100% gain to recover. SPY's 55% drawdown in 2007–09 needed +123% and took until 2012 to recover.
- **One-year risk on $100,000 at 95%:** GBM gives a VaR of **$19,916** and a CVaR of **$25,554**. The iid bootstrap gives **$20,260** and **$26,466**. Actual one-year windows give a CVaR of **$33,243**, because both simulations lose the volatility clustering that makes real bad years so bad.

[writeup.md](writeup.md) has the answers. `HW9_writeup.pdf` is the same document with the plots, for submission.

## Run it

Download 20 years of SPY, save it to `data/`, and run Q1–Q9:

```bash
python hw09_starter.py
```

Rerun on the saved data instead:

```bash
python hw09_starter.py --snapshot
```

Run the 9 self-checks:

```bash
python check_hw09.py
```

You need `numpy`, `pandas`, `scipy`, `matplotlib` and `yfinance`. `_check.py` is the same stand-in helper as in HW4, HW7 and HW8.

## Files

| File | What it is |
|------|------------|
| `hw09_starter.py` | the functions, data download, and the Q1–Q9 output and plots |
| `check_hw09.py`, `_check.py` | self-checks (9/9 pass) |
| `hw09_output.txt` | printed output of the run |
| `data/` | saved SPY closes (raw and adjusted) and run metadata |
| `writeup.md`, `HW9_writeup.pdf` | the answers, with the plots in the PDF |
| `q3_rolling_vol.png` | rolling 60-day volatility, with the spikes labelled |
| `q4_return_histogram.png` | daily returns against a fitted normal, on linear and log scales |
| `q6_recovery.png` | the loss against the gain needed to recover |
| `q7_q9_monte_carlo.png` | final-value distributions and the left tail for GBM, the bootstrap, the block bootstrap and history |
