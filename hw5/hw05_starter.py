"""Homework 5 starter — Algory QI Education, Fall 2026

Reading a real option chain.

    python hw05_starter.py              # pull a live chain from yfinance
    python hw05_starter.py --snapshot   # rerun on the saved CSVs instead

Every live run saves the chains it used to data/, so the numbers in the
write-up can be reproduced after the market has moved on.
"""
import pathlib
import sys
from datetime import date

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"

TICKER = "SPY"
SHORT_TARGET_DAYS = 30     # "roughly one month out"
LONG_MIN_DAYS = 182        # "at least six months out"
MAX_QUOTE_AGE_DAYS = 7     # drop contracts with no trade in the last week
KINDS = ("long_call", "long_put", "short_call", "short_put")


# ---------------------------------------------------------------- Q1
def payoff(kind, strike, premium, spot):
    """Profit per share at expiry of one option position.

    A long position pays the premium up front and receives the option's value
    at expiry. A short position is the other side of the same trade, so its
    profit is exactly the negative of the long one. `spot` may be a number or
    an array; the result has the same shape.
    """
    spot = np.asarray(spot, dtype=float)
    if kind in ("long_call", "short_call"):
        value = np.maximum(spot - strike, 0.0)
    elif kind in ("long_put", "short_put"):
        value = np.maximum(strike - spot, 0.0)
    else:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    profit = value - premium
    if kind.startswith("short"):
        profit = -profit
    return float(profit) if profit.ndim == 0 else profit


def q1():
    print("Q1  payoffs at expiry, strike 100, premium 6")
    print(f"    {'kind':<11}" + "".join(f"{f'spot {s}':>10}" for s in (80, 100, 120)))
    for kind in KINDS:
        row = "".join(f"{payoff(kind, 100, 6, s):>10.2f}" for s in (80, 100, 120))
        print(f"    {kind:<11}{row}")


# ---------------------------------------------------------------- Q2
def q2(strike=100, premium=6):
    spots = np.linspace(50, 150, 401)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    for ax, kind in zip(axes.flat, KINDS):
        ax.plot(spots, payoff(kind, strike, premium, spots), lw=2, color="#2b6cb0")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(strike, color="#c05621", ls="--", lw=1, label=f"strike {strike}")
        ax.set_title(kind.replace("_", " "))
        ax.set_ylabel("profit per share at expiry")
        ax.set_ylim(-55, 55)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper center", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("spot at expiry")
    fig.suptitle(f"Option payoffs, strike {strike}, premium {premium}")
    fig.tight_layout()
    fig.savefig(HERE / "q2_payoffs.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- Q3
def pick_expiry(expiries, today, target_days=None, min_days=None):
    """Choose an expiry from yfinance's list of 'YYYY-MM-DD' strings.

    With `target_days`, return the expiry closest to that many days out.
    With `min_days`, return the first expiry at least that far out.
    """
    today = pd.Timestamp(today)
    days = pd.Series([(pd.Timestamp(e) - today).days for e in expiries], index=expiries)
    if min_days is not None:
        far = days[days >= min_days]
        if far.empty:
            raise ValueError(f"no expiry at least {min_days} days out")
        return far.index[0]
    return (days - target_days).abs().idxmin()


def load_chain(expiry, snapshot=False):
    """Calls for one expiry plus the spot price they were quoted against.

    Live runs pull from yfinance and save a CSV; snapshot runs read it back.
    Spot is stored in every row so a saved chain carries its own spot, which
    is what keeps the two expiries on the same underlying on the same day.
    """
    path = DATA / f"{TICKER}_{expiry}_calls.csv"
    if snapshot:
        calls = pd.read_csv(path)
        return calls, float(calls["spot"].iloc[0])
    import yfinance as yf
    t = yf.Ticker(TICKER)
    spot = float(t.history(period="5d")["Close"].iloc[-1])
    chain = t.option_chain(expiry)
    calls = chain.calls.copy()
    calls["spot"] = spot
    DATA.mkdir(exist_ok=True)
    calls.to_csv(path, index=False)
    return calls, spot, len(chain.calls), len(chain.puts)


# ---------------------------------------------------------------- Q4
def split_call_values(calls, spot, asof=None, max_age_days=MAX_QUOTE_AGE_DAYS):
    """Mid, intrinsic and extrinsic value of every call that has a bid.

    A zero bid means nobody is willing to buy, so the 'mid' would be half the
    ask, a price no one is actually paying. Those rows are dropped.

    With `asof`, contracts that have not traded in the last `max_age_days` are
    dropped too. Outside market hours yfinance serves the bid and ask frozen at
    each contract's last trade, so a deep in-the-money strike last touched in
    March is quoted against March's spot. Measured against today's spot it shows
    a mid far below intrinsic, which an American call can never have.
    """
    keep = calls["bid"] > 0
    if asof is not None and "lastTradeDate" in calls:
        traded = pd.to_datetime(calls["lastTradeDate"], utc=True).dt.tz_localize(None)
        keep &= traded >= pd.Timestamp(asof) - pd.Timedelta(days=max_age_days)
    df = calls.loc[keep, ["strike", "bid", "ask"]].copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["intrinsic"] = np.maximum(spot - df["strike"], 0.0)
    df["extrinsic"] = df["mid"] - df["intrinsic"]
    return df.sort_values("strike").reset_index(drop=True)


def report_drops(calls, values, asof):
    no_bid = int((calls["bid"] <= 0).sum())
    with_bid = split_call_values(calls, 0.0)
    stale = len(with_bid) - len(values)
    print(f"    dropped {no_bid} calls with a zero bid and {stale} with no trade in "
          f"{MAX_QUOTE_AGE_DAYS} days before {asof}; {len(values)} remain")


def nearest_strikes(values, spot, n=10):
    """The n rows whose strike is closest to spot, in strike order."""
    idx = (values["strike"] - spot).abs().nsmallest(n).index
    return values.loc[idx].sort_values("strike")


def print_table(values, spot, label):
    near = nearest_strikes(values, spot)
    print(f"    ten strikes nearest spot ({label})")
    print(f"    {'strike':>8}{'mid':>10}{'intrinsic':>11}{'extrinsic':>11}")
    for r in near.itertuples():
        print(f"    {r.strike:>8.1f}{r.mid:>10.2f}{r.intrinsic:>11.2f}{r.extrinsic:>11.2f}")


def peak(values):
    return values.loc[values["extrinsic"].idxmax()]


# ---------------------------------------------------------------- Q5, Q6
def plot_extrinsic(curves, spot, filename, title):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2b6cb0", "#c05621"]
    for (label, values), color in zip(curves, colors):
        ax.plot(values["strike"], values["extrinsic"], lw=1.6, color=color, label=label)
    ax.axvline(spot, color="black", ls="--", lw=1, label=f"spot {spot:.2f}")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("strike")
    ax.set_ylabel("extrinsic value per share ($)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / filename, dpi=150)
    plt.close(fig)


def above_everywhere(short, long):
    """Fraction of shared strikes where the long-dated extrinsic is higher."""
    both = short.merge(long, on="strike", suffixes=("_s", "_l"))
    return (both["extrinsic_l"] > both["extrinsic_s"]).mean(), len(both)


def main(snapshot=False):
    q1()
    q2()

    print("\nQ3  a live chain")
    if snapshot:
        saved = sorted(DATA.glob(f"{TICKER}_*_calls.csv"))
        short_exp, long_exp = [p.name.split("_")[1] for p in saved]
        short_calls, spot = load_chain(short_exp, snapshot=True)
        long_calls, long_spot = load_chain(long_exp, snapshot=True)
        today = pd.read_csv(DATA / "run_date.txt", header=None).iloc[0, 0]
        n_calls, n_puts = len(short_calls), None
    else:
        import yfinance as yf
        today = date.today().isoformat()
        expiries = yf.Ticker(TICKER).options
        print(f"    available expiries: {', '.join(expiries)}")
        short_exp = pick_expiry(expiries, today, target_days=SHORT_TARGET_DAYS)
        long_exp = pick_expiry(expiries, today, min_days=LONG_MIN_DAYS)
        short_calls, spot, n_calls, n_puts = load_chain(short_exp)
        long_calls, long_spot, _, _ = load_chain(long_exp)
        (DATA / "run_date.txt").write_text(today + "\n")
    days_short = (pd.Timestamp(short_exp) - pd.Timestamp(today)).days
    days_long = (pd.Timestamp(long_exp) - pd.Timestamp(today)).days
    print(f"    ticker        {TICKER}")
    print(f"    run date      {today}")
    print(f"    expiry        {short_exp}  ({days_short} days out)")
    print(f"    spot          {spot:.2f}")
    print(f"    calls         {n_calls}")
    if n_puts is not None:
        print(f"    puts          {n_puts}")
    assert spot == long_spot, "the two chains must be quoted against the same spot"

    print("\nQ4  intrinsic and extrinsic value")
    short_vals = split_call_values(short_calls, spot, asof=today)
    report_drops(short_calls, short_vals, today)
    print_table(short_vals, spot, short_exp)
    worst = (short_vals["intrinsic"] + short_vals["extrinsic"] - short_vals["mid"]).abs().max()
    print(f"    max |intrinsic + extrinsic - mid| = {worst:.2e}")

    print("\nQ5  where extrinsic value peaks")
    plot_extrinsic([(f"{short_exp} ({days_short}d)", short_vals)], spot,
                   "q5_extrinsic_short.png", f"{TICKER} call extrinsic value, {short_exp}")
    p = peak(short_vals)
    print(f"    largest extrinsic  {p.extrinsic:.2f} at strike {p.strike:.1f}")
    print(f"    spot               {spot:.2f}")
    deep = short_vals.iloc[0]
    print(f"    deepest in-the-money strike {deep.strike:.1f}: extrinsic {deep.extrinsic:.2f}")

    print(f"\nQ6  a longer-dated expiry: {long_exp} ({days_long} days out)")
    long_vals = split_call_values(long_calls, spot, asof=today)
    report_drops(long_calls, long_vals, today)
    print_table(long_vals, spot, long_exp)
    plot_extrinsic([(f"{short_exp} ({days_short}d)", short_vals),
                    (f"{long_exp} ({days_long}d)", long_vals)], spot,
                   "q6_extrinsic_both.png",
                   f"{TICKER} call extrinsic value, short vs long expiry")
    pl = peak(long_vals)
    print(f"    largest extrinsic, {short_exp}: {p.extrinsic:.2f} at strike {p.strike:.1f}")
    print(f"    largest extrinsic, {long_exp}: {pl.extrinsic:.2f} at strike {pl.strike:.1f}")
    frac, n = above_everywhere(short_vals, long_vals)
    print(f"    long-dated curve is higher at {frac:.0%} of the {n} strikes both chains quote")


if __name__ == "__main__":
    main(snapshot="--snapshot" in sys.argv)
