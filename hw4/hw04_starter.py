"""Homework 4 starter — Algory QI Education, Fall 2026

The backtester is the piece worth getting right. Everything after it in the
semester assumes it works.
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

LOOKBACK = 60
TRADING_DAYS = 252

# Flip to True only once the final strategy is chosen from training results.
# The test set is run once, in Q8, and nothing is changed after seeing it.
RUN_TEST = True


# ---------------------------------------------------------------- Q1
def download_prices(start="1990-01-01"):
    """Daily closes of ^SP500TR, the total-return S&P 500, since `start`."""
    close = yf.download("^SP500TR", start=start, auto_adjust=True, progress=False)["Close"]
    return close.squeeze().dropna()


def load_and_split(split_date="2011-09-09", prices=None, warmup=LOOKBACK):
    """Split ^SP500TR since 1990 into training and test at `split_date`.

    Training ends on the split date's close. Test begins `warmup` trading days
    BEFORE it, so the 60-day window is already full on the first day traded out
    of sample. Those overlapping days are history the strategy reads, never days
    it earns a return on. Cut the series at one date with no overlap and the
    first 60 days of the test set have no window, so the out-of-sample period
    silently starts three months late.

    September 9, 2011 is a Friday. A split date has to be a trading day, or
    "ends on that close" names a close that does not exist.

    Pass `prices` to split a series you already have, which is how the checker
    tests this without a download. Return (training, test).
    """
    if prices is None:
        prices = download_prices()
    split = pd.Timestamp(split_date)
    if split not in prices.index:
        raise ValueError(f"{split_date} is not a trading day in the data")
    pos = prices.index.get_loc(split)
    training = prices.iloc[:pos + 1]
    # the split close is the last day of the warm-up, so the first return the
    # test backtest earns runs from the split close to the next session
    test = prices.iloc[pos - warmup + 1:]
    return training, test


# ---------------------------------------------------------------- Q2, Q4, Q5
def always_hold(window):
    """Always invest. The control case, and the one that proves the backtester."""
    return True


def always_cash(window):
    """Never invest. Must produce exactly 0% return."""
    return False


def make_random_hold(p=0.7, seed=0):
    """Return a strategy that invests with probability p each day.

    A factory, not a plain function, because the generator has to live somewhere
    and a default argument is the wrong place: `rng=np.random.default_rng(0)` in
    a signature is evaluated once when the module loads, so every backtest in the
    session shares one advancing generator and the same backtest returns a
    different number each time you run it. Building the generator here gives one
    per strategy, so `backtest(train, make_random_hold())` is reproducible.
    """
    rng = np.random.default_rng(seed)

    def random_hold(window):
        return bool(rng.random() < p)

    return random_hold


def five_day_reversal(window):
    """The rule from class: hold unless the last 5 days were positive.

    Measure that return from the close 5 days before today to today's close,
    both of which are inside the window.
    """
    five_day_return = window[-1] / window[-6] - 1
    return bool(not five_day_return > 0)


# ---------------------------------------------------------------- Q6 helpers
def sma(window, n):
    """Simple moving average of the last n closes in the window, today included."""
    return float(np.mean(window[-n:]))


def ema(window, n):
    """Exponential moving average over the window, span n, seeded at the oldest close."""
    alpha = 2 / (n + 1)
    value = window[0]
    for price in window[1:]:
        value = alpha * price + (1 - alpha) * value
    return float(value)


def price_above_sma(n, band=0.0):
    """Hold if today's close is above (1 + band) times its n-day average."""
    def rule(window):
        return bool(window[-1] > (1 + band) * sma(window, n))
    return rule


def price_above_ema(n):
    """Hold if today's close is above its n-span exponential average."""
    def rule(window):
        return bool(window[-1] > ema(window, n))
    return rule


def sma_crossover(short, long):
    """Hold if the short average is above the long average."""
    def rule(window):
        return bool(sma(window, short) > sma(window, long))
    return rule


def sma_rising(n, lag):
    """Hold if the n-day average today is above the n-day average `lag` days ago.

    Both averages fit inside the window only when n + lag <= LOOKBACK.
    """
    def rule(window):
        return bool(sma(window, n) > sma(window[:-lag], n))
    return rule


def price_and_cross(n_price, short, long):
    """Hold if the close is above its n_price-day average AND short > long."""
    above = price_above_sma(n_price)
    cross = sma_crossover(short, long)

    def rule(window):
        return above(window) and cross(window)
    return rule


# Every variant tried, in the order tried. All of them are printed in Q6.
VARIANTS = [
    ("close > 10-day SMA",                       price_above_sma(10)),
    ("close > 20-day SMA",                       price_above_sma(20)),
    ("close > 30-day SMA",                       price_above_sma(30)),
    ("close > 40-day SMA",                       price_above_sma(40)),
    ("close > 50-day SMA",                       price_above_sma(50)),
    ("close > 60-day SMA",                       price_above_sma(60)),
    ("close > 0.98 x 60-day SMA (2% band)",      price_above_sma(60, band=-0.02)),
    ("close > 1.02 x 60-day SMA (2% band)",      price_above_sma(60, band=0.02)),
    ("close > 20-span EMA",                      price_above_ema(20)),
    ("close > 40-span EMA",                      price_above_ema(40)),
    ("5-day SMA > 20-day SMA",                   sma_crossover(5, 20)),
    ("10-day SMA > 30-day SMA",                  sma_crossover(10, 30)),
    ("10-day SMA > 50-day SMA",                  sma_crossover(10, 50)),
    ("20-day SMA > 60-day SMA",                  sma_crossover(20, 60)),
    ("20-day SMA rising over 10 days",           sma_rising(20, 10)),
    ("40-day SMA rising over 20 days",           sma_rising(40, 20)),
    ("close > 50-day SMA AND 10 > 50-day SMA",   price_and_cross(50, 10, 50)),
]

# Selection rule, fixed before any variant was run: the final strategy is the
# variant with the highest training-set annualised return. No test-set peeking.
FINAL_RULE = "40-day SMA rising over 20 days"   # the Q6 training winner


def my_strategy(window):
    """Your own rule. It sees only `window`, the previous LOOKBACK closes.

    It must never look at anything after that. That is the whole assignment.

    Hold the index when today's 40-day simple moving average is above the 40-day
    average as it stood 20 trading days ago, i.e. the medium-term trend is up.
    Otherwise hold cash. Uses closes 0..59 of the window and nothing else.
    """
    return sma_rising(40, 20)(window)


# ---------------------------------------------------------------- Q3
def backtest(prices, strategy, lookback=LOOKBACK):
    """Walk forward one day at a time and apply the strategy's decision.

    The convention, stated exactly, because everything else depends on it:

        on day i, the strategy is handed prices[i - lookback + 1 : i + 1],
        the `lookback` closes ENDING WITH today's, today's included;
        if it says hold you earn prices[i + 1] / prices[i] - 1, otherwise zero.

    So i runs from lookback - 1 to len(prices) - 2. The first decision is made on
    the 60th close and the first return earned runs from the 60th close to the
    61st. Today's close is known when you decide, so withholding it throws away a
    day of information for nothing; tomorrow's close is what you are predicting,
    so reading it is the bug this whole session is about.

    Return a dict with at least:
        total_return, annualised_return, max_drawdown, fraction_invested

    Annualise by trading days, 252 a year, compounded: value ** (252 / days) - 1.

    Sanity checks before you trust it:
        always_hold  must match buy-and-hold over the days actually traded,
                     prices[-1] / prices[lookback - 1] - 1, not over the
                     whole file
        always_cash  must return exactly 0.0
    If either fails you have an off-by-one, and every number after it is fiction.
    """
    p = np.asarray(prices, dtype=float)
    days = len(p) - lookback          # number of returns applied
    if days <= 0:
        raise ValueError("need more than `lookback` prices to trade at all")

    values = np.empty(days + 1)
    values[0] = 1.0
    invested = 0
    value = 1.0
    for k, i in enumerate(range(lookback - 1, len(p) - 1)):
        window = p[i - lookback + 1:i + 1]       # ends with today's close, p[i]
        if strategy(window):
            value *= p[i + 1] / p[i]             # tomorrow's return, never shown to the strategy
            invested += 1
        values[k + 1] = value

    running_peak = np.maximum.accumulate(values)
    return {
        "total_return": value - 1.0,
        "annualised_return": value ** (TRADING_DAYS / days) - 1.0,
        "max_drawdown": float(np.min(values / running_peak - 1.0)),
        "fraction_invested": invested / days,
        "days": days,
        "values": values,
    }


# ---------------------------------------------------------------- output
def plot_rebased(prices, results, title, filename, lookback=LOOKBACK):
    """Plot each strategy's portfolio value, rebased to 1 on the first trading day."""
    dates = prices.index[lookback - 1:]
    plt.figure(figsize=(9, 4.5))
    for label, r in results.items():
        # dashed index line, so it stays visible when it sits exactly on a strategy
        style = "--" if label.startswith("index") else "-"
        plt.plot(dates, r["values"], style, label=label)
    plt.yscale("log")
    plt.ylabel("value of $1 (log scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def describe(name, r):
    return (f"{name:<42} total {r['total_return']:+9.1%}   "
            f"annual {r['annualised_return']:+6.2%}")


def main():
    # ------------------------------------------------------------ Q1
    full = download_prices()
    train, test = load_and_split(prices=full)
    print("Q1  data: ^SP500TR daily closes")
    print(f"    full    {len(full):>5} days  {full.index[0].date()} to {full.index[-1].date()}")
    print(f"    train   {len(train):>5} days  {train.index[0].date()} to {train.index[-1].date()}"
          f"   mean price {train.mean():9.2f}")
    print(f"    test    {len(test):>5} days  {test.index[0].date()} to {test.index[-1].date()}"
          f"   mean price {test.mean():9.2f}")
    print(f"            ({LOOKBACK} of those are warm-up history; "
          f"{len(test) - LOOKBACK} out-of-sample days, first return earned after "
          f"{test.index[LOOKBACK - 1].date()})")

    # ------------------------------------------------------------ Q3, Q4, Q5 on training
    print("\nQ3-Q5  training set")
    hold = backtest(train, always_hold)
    index_total = train.iloc[-1] / train.iloc[LOOKBACK - 1] - 1
    controls = [("always hold", always_hold),
                ("always cash", always_cash),
                ("random 70% (seed 0)", make_random_hold(0.7, seed=0)),
                ("5-day reversal", five_day_reversal)]
    control_results = {}
    for name, strat in controls:
        r = backtest(train, strat)
        control_results[name] = r
        print("    " + describe(name, r))
    print(f"    index buy-and-hold over the same days: total {index_total:+.1%}, "
          f"always-hold differs by {hold['total_return'] - index_total:+.2e}")
    full_index = backtest(full, always_hold)
    print(f"    index over the full period: annual {full_index['annualised_return']:+.2%}")
    plot_rebased(train, {"always-hold strategy": hold,
                         "index (buy and hold)": {"values": train.iloc[LOOKBACK - 1:].to_numpy()
                                                  / train.iloc[LOOKBACK - 1]}},
                 "Training set: always-hold strategy vs index, rebased to 1",
                 "q3_always_hold_vs_index.png")
    plot_rebased(train, {k: control_results[k] for k in
                         ["always hold", "always cash", "random 70% (seed 0)", "5-day reversal"]},
                 "Training set: control strategies and the class rule",
                 "q4_q5_controls_training.png")

    # ------------------------------------------------------------ Q6
    print(f"\nQ6  every variant tried ({len(VARIANTS)}), training-set annualised return")
    print(f"    {'index (always hold)':<42} annual {hold['annualised_return']:+6.2%}")
    variant_results = {}
    for name, strat in VARIANTS:
        r = backtest(train, strat)
        variant_results[name] = r
        print(f"    {name:<42} annual {r['annualised_return']:+6.2%}   "
              f"invested {r['fraction_invested']:5.1%}   max DD {r['max_drawdown']:+6.1%}")
    best = max(variant_results, key=lambda k: variant_results[k]["annualised_return"])
    print(f"    best by the pre-stated rule (highest training annualised return): {best}")

    if FINAL_RULE is None:
        print("\nFINAL_RULE not set yet: stopping before Q7 and the test set.")
        return

    # ------------------------------------------------------------ Q7
    print(f"\nQ7  final rule: hold the index tomorrow if today's 40-day simple moving average "
          f"is above the 40-day average from 20 trading days ago; otherwise hold cash.")
    if best != FINAL_RULE:
        print(f"    WARNING: FINAL_RULE ({FINAL_RULE}) is not the training winner ({best})")
    mine_train = backtest(train, my_strategy)
    print("    " + describe("final rule, training", mine_train)
          + f"   max DD {mine_train['max_drawdown']:+.1%}"
            f"   invested {mine_train['fraction_invested']:.1%}")
    print("    " + describe("index, training", hold) + f"   max DD {hold['max_drawdown']:+.1%}")
    plot_rebased(train, {"final rule": mine_train, "index": hold},
                 "Training set: final rule vs index", "q7_final_rule_training.png")

    if not RUN_TEST:
        print("\nRUN_TEST is False: the test set has not been touched.")
        return

    # ------------------------------------------------------------ Q8: once
    print("\nQ8  TEST SET, run once")
    mine_test = backtest(test, my_strategy)
    index_test = backtest(test, always_hold)
    for name, r in [("final rule", mine_test), ("index", index_test)]:
        print(f"    {name:<12} total {r['total_return']:+8.1%}   annual {r['annualised_return']:+6.2%}"
              f"   max DD {r['max_drawdown']:+6.1%}   invested {r['fraction_invested']:6.1%}")
    plot_rebased(test, {"final rule": mine_test, "index": index_test},
                 "Test set (out of sample): final rule vs index", "q8_final_rule_test.png")

    # ------------------------------------------------------------ Q9
    window = full.to_numpy()[-LOOKBACK:]
    decision = "HOLD the index" if my_strategy(window) else "CASH"
    print(f"\nQ9  today's call: {decision}  (last close used: {full.index[-1].date()}, "
          f"40-day SMA today {sma(window, 40):.2f} vs 20 days ago {sma(window[:-20], 40):.2f})")


if __name__ == "__main__":
    main()
