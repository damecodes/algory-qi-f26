# HW7: machine learning on next-day SPY returns

This folder holds:

- a feature table of 15 years of daily SPY data (seven features built only from past data, with the next day's return as the label);
- a time-ordered 60/20/20 split into training, validation and test;
- four scikit-learn models: linear regression, ridge, a random forest and gradient boosting;
- one hyperparameter tuned for each model on validation, and the best model run once on the test set;
- a hold-or-cash trading rule backtested with the HW4 backtester;
- a robustness study of how much of the result is luck.

## Result

The run used data to the Friday 2026-11-06 close (3,771 trading days from 2011-11-08).

- **Overfitting:** the default random forest has a training R² of **0.85** and a validation R² of **−0.045**. All four models are negative on validation.
- **Best model:** a random forest with `min_samples_leaf=50` (validation R² **+0.0038**). On the test set it scored R² **+0.0067** and a correlation of **+0.088**, mostly from a few days in April 2025. Its rank correlation is 0.047, which is not significant.
- **Trading rule:** **+17.7% a year**, against **+19.3%** for buy-and-hold over 2023-11 to 2026-11. It made money but lost to buy-and-hold, because the days it sat out still rose on average.
- **Robustness:** changing only the forest's random seed moves the rule's return from +13.6% to +18.8% a year. It never beats buy-and-hold in 40 refits.

See [writeup.md](writeup.md) for the answers. Q8 explains why an R² near zero is the expected result, and why a rule can still make money with it.

## Run it

Download 15 years of SPY and T-bill rates, save them to `data/`, and run Q1–Q7:

```bash
python hw07_starter.py
```

Rerun on the saved data instead:

```bash
python hw07_starter.py --snapshot
```

Refit the final model under 20 random seeds and 20 draws of one-in-a-million price noise (about 20 seconds):

```bash
python robustness.py
```

Run the 5 self-checks:

```bash
python check_hw07.py
```

You need `numpy`, `pandas`, `matplotlib`, `yfinance` and `scikit-learn`. Q6 imports `backtest()` from `../hw4/hw04_starter.py`, so keep the two folders side by side. `_check.py` is the same stand-in helper as in HW4.

## Files

| File | What it is |
|------|------------|
| `hw07_starter.py` | features, split, models, tuning, test run, backtest, and the Q1–Q7 output |
| `robustness.py` | refits the chosen model under different seeds and price noise |
| `check_hw07.py`, `_check.py` | self-checks (5/5 pass) |
| `hw07_output.txt`, `robustness_output.txt` | printed output of the runs |
| `data/` | saved SPY closes and volume, 13-week T-bill rates (`^IRX`), and run metadata |
| `writeup.md`, `HW7_writeup.pdf` | the answers, with Q8 on its own page |
| `q4_tuning_curves.png`, `q6_test_backtest.png`, `q7_feature_importance.png` | plots |
