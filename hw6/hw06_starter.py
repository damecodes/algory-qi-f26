"""Homework 6 starter — Algory QI Education, Fall 2026

Greeks, combinations, and implied volatility.

    python hw06_starter.py              # pull live SPY data from yfinance
    python hw06_starter.py --snapshot   # rerun on the saved data/ files

Every live run saves what it used to data/, so the numbers in the write-up can
be reproduced after the market has moved on.
"""
import json
import pathlib
import sys
from datetime import date
from math import log, sqrt, exp, erf

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"

TICKER = "SPY"
TARGET_DAYS = 30           # the VIX horizon, and the realised-vol window
MAX_QUOTE_AGE_DAYS = 7     # drop contracts with no trade in the last week
RV_WINDOW = 30             # trading days of returns for realised volatility
TRADING_DAYS = 252
SIM_START = "2005-01-01"   # Q7 short-straddle history


# ---------------------------------------------------------------- Q3
def norm_cdf(x):
    """Standard normal CDF from math.erf."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes(kind, spot, strike, years, rate, vol):
    """European option price.

    Time in YEARS. Rate and vol as decimals, not percentages. That is the usual bug.
    black_scholes("call", 100, 100, 1, 0.04, 0.25) -> 11.84
    black_scholes("put",  100, 100, 1, 0.04, 0.25) ->  7.92
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', not {kind!r}")
    discounted_strike = strike * exp(-rate * years)
    if years <= 0 or vol <= 0:
        # no time or no uncertainty left: the option is worth its discounted intrinsic
        if kind == "call":
            return max(spot - discounted_strike, 0.0)
        return max(discounted_strike - spot, 0.0)
    d1 = (log(spot / strike) + (rate + 0.5 * vol ** 2) * years) / (vol * sqrt(years))
    d2 = d1 - vol * sqrt(years)
    if kind == "call":
        return spot * norm_cdf(d1) - discounted_strike * norm_cdf(d2)
    return discounted_strike * norm_cdf(-d2) - spot * norm_cdf(-d1)


# ---------------------------------------------------------------- Q4
def implied_volatility(price, spot, strike, years, rate, kind, lo=0.0001, hi=5.0, tol=1e-8):
    """Solve for the volatility that makes black_scholes match `price`.

    Bisection between about 0.0001 and 5.0 is plenty and is easier to debug than
    Newton's method. Price rises with volatility, which is what makes it work.
    Feeding your own price back in must return the volatility you started with.

    Returns nan when no volatility in [lo, hi] reproduces the price: a quote
    below the zero-volatility value, or above the 500% one, has no answer.
    """
    if not (black_scholes(kind, spot, strike, years, rate, lo) <= price
            <= black_scholes(kind, spot, strike, years, rate, hi)):
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if black_scholes(kind, spot, strike, years, rate, mid) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- Q1
def payoff(kind, strike, premium, spot):
    """Profit per share at expiry of one option position (from HW5).

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
        raise ValueError(f"unknown option kind {kind!r}")
    profit = value - premium
    if kind.startswith("short"):
        profit = -profit
    return float(profit) if profit.ndim == 0 else profit


def shares(entry, spot):
    """Profit per share of stock bought at `entry`."""
    profit = np.asarray(spot, dtype=float) - entry
    return float(profit) if profit.ndim == 0 else profit


def covered_call(entry, strike, premium, spot):
    """Long 100 shares bought at `entry`, short one call struck at `strike`.

    Returns profit per share at expiry. Above the strike this must stop rising.
    """
    return shares(entry, spot) + payoff("short_call", strike, premium, spot)


def protective_put(entry, strike, premium, spot):
    """Long 100 shares bought at `entry`, long one put struck at `strike`."""
    return shares(entry, spot) + payoff("long_put", strike, premium, spot)


def long_straddle(strike, call_premium, put_premium, spot):
    """Long a call and a put at the same strike."""
    return payoff("long_call", strike, call_premium, spot) + payoff("long_put", strike, put_premium, spot)


def bull_call_spread(low_strike, low_premium, high_strike, high_premium, spot):
    """Long a call at the lower strike, short a call at the higher strike."""
    return (payoff("long_call", low_strike, low_premium, spot)
            + payoff("short_call", high_strike, high_premium, spot))


def summarise(profit, kinks):
    """Max profit, max loss and break-evens of a piecewise-linear expiry profit.

    Between kinks (strikes) the profit is a straight line, so its extremes are
    at spot 0, at a kink, or at infinity, and each break-even is found exactly by
    interpolating between two points that straddle zero. Unbounded ends are
    detected from the slope beyond the last kink.
    """
    kinks = sorted(kinks)
    xs = [0.0] + kinks
    ys = [float(profit(x)) for x in xs]
    far_slope = float(profit(kinks[-1] + 1.0)) - ys[-1]

    max_profit = float("inf") if far_slope > 1e-12 else max(ys)
    max_loss = float("-inf") if far_slope < -1e-12 else min(ys)

    breakevens = []
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 == 0.0:
            breakevens.append(x0)
        elif y0 * y1 < 0:
            breakevens.append(x0 + (x1 - x0) * (-y0) / (y1 - y0))
    if ys[-1] == 0.0:
        breakevens.append(xs[-1])
    elif ys[-1] * far_slope < 0:                 # crosses zero beyond the last kink
        breakevens.append(xs[-1] - ys[-1] / far_slope)
    return max_profit, max_loss, breakevens


# ---------------------------------------------------------------- market data
def pick_expiry(expiries, today, target_days=TARGET_DAYS):
    """The expiry closest to `target_days` calendar days after `today`."""
    today = pd.Timestamp(today)
    days = pd.Series([(pd.Timestamp(e) - today).days for e in expiries], index=expiries)
    return (days - target_days).abs().idxmin()


def last_close(ticker, period="10d"):
    hist = yf.Ticker(ticker).history(period=period)["Close"].dropna()
    return float(hist.iloc[-1]), hist.index[-1].date().isoformat()


def load_market(snapshot=False):
    """Everything the live questions need, from yfinance or from data/.

    Returns a dict: run_date, quote_date (the last close the quotes belong to),
    spot, rate (13-week T-bill, decimal), vix, expiry, calls, puts, closes.
    """
    meta_path = DATA / "meta.json"
    if snapshot:
        meta = json.loads(meta_path.read_text())
    else:
        t = yf.Ticker(TICKER)
        closes = t.history(period="6mo", auto_adjust=False)["Close"].dropna()
        closes.index = closes.index.tz_localize(None).normalize()
        quote_date = closes.index[-1].date().isoformat()
        expiry = pick_expiry(t.options, quote_date)
        chain = t.option_chain(expiry)
        irx, irx_date = last_close("^IRX")
        vix, vix_date = last_close("^VIX")
        meta = {"run_date": date.today().isoformat(), "quote_date": quote_date,
                "spot": float(closes.iloc[-1]), "rate": irx / 100.0, "rate_date": irx_date,
                "vix": vix, "vix_date": vix_date, "expiry": expiry}
        DATA.mkdir(exist_ok=True)
        chain.calls.to_csv(DATA / f"{TICKER}_{expiry}_calls.csv", index=False)
        chain.puts.to_csv(DATA / f"{TICKER}_{expiry}_puts.csv", index=False)
        closes.rename("close").to_csv(DATA / f"{TICKER}_closes.csv")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    meta["calls"] = pd.read_csv(DATA / f"{TICKER}_{meta['expiry']}_calls.csv")
    meta["puts"] = pd.read_csv(DATA / f"{TICKER}_{meta['expiry']}_puts.csv")
    meta["closes"] = pd.read_csv(DATA / f"{TICKER}_closes.csv", index_col=0, parse_dates=True)["close"]
    meta["years"] = (pd.Timestamp(meta["expiry"]) - pd.Timestamp(meta["quote_date"])).days / 365.0
    return meta


def clean_quotes(chain, asof, max_age_days=MAX_QUOTE_AGE_DAYS):
    """Contracts with a non-zero bid and a trade in the last week, with a mid.

    A zero bid means nobody is willing to buy, so its 'mid' is not a price. A
    contract with no recent trade carries a bid and ask frozen at its last trade
    (the HW5 problem), which is quoted against a spot that no longer exists.
    """
    keep = chain["bid"] > 0
    if "lastTradeDate" in chain:
        traded = pd.to_datetime(chain["lastTradeDate"], utc=True).dt.tz_localize(None)
        keep &= traded >= pd.Timestamp(asof) - pd.Timedelta(days=max_age_days)
    df = chain.loc[keep, ["strike", "bid", "ask"]].copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    return df.sort_values("strike").reset_index(drop=True)


def mid_at(quotes, strike):
    return float(quotes.loc[quotes["strike"] == strike, "mid"].iloc[0])


def realised_vol(closes, window=RV_WINDOW):
    """Annualised standard deviation of the last `window` daily log returns."""
    returns = np.log(closes / closes.shift(1)).dropna().iloc[-window:]
    return float(returns.std(ddof=1) * sqrt(TRADING_DAYS)), len(returns)


# ---------------------------------------------------------------- output helpers
def fmt(x):
    if x == float("inf"):
        return "unlimited"
    if x == float("-inf"):
        return "unlimited"
    return f"{x:+.2f}"


def q1(mkt):
    spot = mkt["spot"]
    calls = clean_quotes(mkt["calls"], mkt["quote_date"])
    puts = clean_quotes(mkt["puts"], mkt["quote_date"])
    both = sorted(set(calls["strike"]) & set(puts["strike"]))
    atm = min(both, key=lambda k: abs(k - spot))
    otm_call = min(k for k in calls["strike"] if k >= spot * 1.02)    # about 2% above spot
    otm_put = max(k for k in puts["strike"] if k <= spot * 0.98)      # about 2% below spot

    c_atm, p_atm = mid_at(calls, atm), mid_at(puts, atm)
    c_otm, p_otm = mid_at(calls, otm_call), mid_at(puts, otm_put)

    positions = [
        ("covered call", f"long stock at {spot:.2f} + short {otm_call:g} call at {c_otm:.2f}",
         lambda s: covered_call(spot, otm_call, c_otm, s), [otm_call]),
        ("protective put", f"long stock at {spot:.2f} + long {otm_put:g} put at {p_otm:.2f}",
         lambda s: protective_put(spot, otm_put, p_otm, s), [otm_put]),
        ("long straddle", f"long {atm:g} call at {c_atm:.2f} + long {atm:g} put at {p_atm:.2f}",
         lambda s: long_straddle(atm, c_atm, p_atm, s), [atm]),
        ("bull call spread", f"long {atm:g} call at {c_atm:.2f} + short {otm_call:g} call at {c_otm:.2f}",
         lambda s: bull_call_spread(atm, c_atm, otm_call, c_otm, s), [atm, otm_call]),
    ]

    print(f"Q1  combinations on live {TICKER} mids, expiry {mkt['expiry']}, spot {spot:.2f}"
          f" (close of {mkt['quote_date']})")
    print("    profit per share at expiry; one contract is 100 shares")
    print(f"    {'position':<17}{'max profit':>12}{'max loss':>12}   break-even(s)")
    spots = np.linspace(spot * 0.85, spot * 1.15, 601)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, legs, fn, kinks) in zip(axes.flat, positions):
        mp, ml, be = summarise(fn, kinks)
        print(f"    {name:<17}{fmt(mp):>12}{fmt(ml):>12}   {', '.join(f'{b:.2f}' for b in be)}")
        print(f"      {legs}")
        ax.plot(spots, fn(spots), lw=2, color="#2b6cb0")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(spot, color="grey", ls=":", lw=1, label=f"spot {spot:.2f}")
        for k in sorted(set(kinks)):
            ax.axvline(k, color="#c05621", ls="--", lw=1)
        for b in be:
            ax.plot(b, 0, "o", color="#2f855a", ms=5)
        ax.set_title(f"{name}\n{legs}", fontsize=9)
        ax.set_xlabel("SPY at expiry")
        ax.set_ylabel("profit per share ($)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Q1: four combinations, live {TICKER} quotes, expiry {mkt['expiry']} "
                 f"(dashed = strikes, dots = break-evens)")
    fig.tight_layout()
    fig.savefig(HERE / "q1_combinations.png", dpi=150)
    plt.close(fig)
    return atm


Q2_TEXT = """\
    An expiry payoff diagram assumes every leg expires on the same day, so the
    position is fully described by where the underlying ends up. In a calendar
    spread the short one-week call expires while the long three-month call still
    has about twelve weeks left, so on that day the long leg is worth its
    Black-Scholes value, which depends on volatility, time and rates, not just on
    the stock price. If the stock does nothing for a week, the short call loses
    almost all of its time value and the long call loses only a little, so the
    spread makes money: it is betting that the underlying stays near the strike
    in the short term (it is short gamma and long theta), and that implied
    volatility for the back month holds up or rises (it is long vega)."""


def q2(mkt, atm_iv, strike):
    """The calendar spread valued on the day the short leg expires, with Black-Scholes."""
    spot, rate = mkt["spot"], mkt["rate"]
    t_long, t_short = 91 / 365, 7 / 365
    long0 = black_scholes("call", spot, strike, t_long, rate, atm_iv)
    short0 = black_scholes("call", spot, strike, t_short, rate, atm_iv)
    debit = long0 - short0
    print(f"\nQ2  calendar spread: long {strike:g} call 3 months out, short {strike:g} call 1 week out")
    print(Q2_TEXT)
    spots = np.linspace(spot * 0.92, spot * 1.08, 401)
    t_left = t_long - t_short
    print(f"    illustration with Black-Scholes at the {atm_iv:.1%} ATM implied vol: net debit {debit:.2f}")
    fig, ax = plt.subplots(figsize=(9, 5))
    for shift, style in [(-0.05, ":"), (0.0, "-"), (0.05, "--")]:
        vol = atm_iv + shift
        pnl = np.array([black_scholes("call", s, strike, t_left, rate, vol)
                        - max(s - strike, 0.0) - debit for s in spots])
        unchanged = black_scholes("call", spot, strike, t_left, rate, vol) - max(spot - strike, 0.0) - debit
        print(f"    one week later, SPY unchanged, implied vol {vol:.1%}: P/L {unchanged:+.2f} per share")
        ax.plot(spots, pnl, style, lw=2, label=f"implied vol {vol:.0%} after a week")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(strike, color="#c05621", ls="--", lw=1, label=f"strike {strike:g}")
    ax.set_xlabel("SPY when the short leg expires")
    ax.set_ylabel("profit per share ($)")
    ax.set_title("Q2: calendar spread on the day the 1-week call expires\n"
                 "the same spot gives different P/L depending on volatility")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "q2_calendar_spread.png", dpi=150)
    plt.close(fig)


def q3():
    c = black_scholes("call", 100, 100, 1, 0.04, 0.25)
    p = black_scholes("put", 100, 100, 1, 0.04, 0.25)
    print("\nQ3  Black-Scholes, spot 100, strike 100, 1 year, 4% rate, 25% vol")
    print(f"    call {c:.2f}   put {p:.2f}")
    print(f"    put-call parity: C - P = {c - p:.4f},  S - K*exp(-rT) = {100 - 100 * exp(-0.04):.4f}")


def q4():
    print("\nQ4  implied volatility round trip (price from my pricer, solved back)")
    for kind, k, vol in [("call", 100, 0.25), ("put", 100, 0.25), ("call", 120, 0.18), ("put", 80, 0.60)]:
        price = black_scholes(kind, 100, k, 1, 0.04, vol)
        iv = implied_volatility(price, 100, k, 1, 0.04, kind)
        print(f"    {kind:<4} strike {k:>3}  vol in {vol:.4f}  price {price:7.4f}  vol out {iv:.4f}")


def smile(quotes, mkt, kind):
    """Implied vol of every quote, solved from its mid. Unsolvable quotes become nan."""
    out = quotes.copy()
    out["iv"] = [implied_volatility(m, mkt["spot"], k, mkt["years"], mkt["rate"], kind)
                 for k, m in zip(out["strike"], out["mid"])]
    return out


def q5(mkt):
    spot = mkt["spot"]
    calls = smile(clean_quotes(mkt["calls"], mkt["quote_date"]), mkt, "call")
    puts = smile(clean_quotes(mkt["puts"], mkt["quote_date"]), mkt, "put")
    solved = calls.dropna(subset=["iv"])
    otm_puts = puts[(puts["strike"] < spot)].dropna(subset=["iv"])
    n_total, n_bid = len(mkt["calls"]), int((mkt["calls"]["bid"] > 0).sum())

    print(f"\nQ5  implied volatility across the {mkt['expiry']} {TICKER} call chain")
    print(f"    quotes as of the {mkt['quote_date']} close, {mkt['years'] * 365:.0f} days to expiry, "
          f"rate {mkt['rate']:.3%} (13-week T-bill, {mkt['rate_date']})")
    print(f"    {n_total} calls, {n_bid} with a non-zero bid, {len(calls)} also traded in the last "
          f"{MAX_QUOTE_AGE_DAYS} days, {len(solved)} with a solvable implied vol")
    atm_row = solved.loc[(solved["strike"] - spot).abs().idxmin()]
    print(f"    implied vol at strike {atm_row.strike:g} (nearest spot {spot:.2f}): {atm_row.iv:.2%}")
    print(f"    VIX close {mkt['vix_date']}: {mkt['vix']:.2f}  (17.1 on 2026-09-14)")
    lo = solved.iloc[(solved["strike"] - spot * 0.90).abs().argmin()]
    hi = solved.iloc[(solved["strike"] - spot * 1.05).abs().argmin()]
    print(f"    implied vol 10% below spot (strike {lo.strike:g}): {lo.iv:.2%}   "
          f"5% above spot (strike {hi.strike:g}): {hi.iv:.2%}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(solved["strike"], solved["iv"] * 100, "o-", ms=3, lw=1.2, color="#2b6cb0",
            label="calls (mid, non-zero bid)")
    ax.plot(otm_puts["strike"], otm_puts["iv"] * 100, "x", ms=4, color="#c05621", alpha=0.7,
            label="out-of-the-money puts, for comparison")
    ax.axvline(spot, color="black", ls="--", lw=1, label=f"spot {spot:.2f}")
    ax.axhline(mkt["vix"], color="#2f855a", ls=":", lw=1.5, label=f"VIX {mkt['vix']:.2f}")
    ax.set_xlabel("strike")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title(f"Q5: {TICKER} implied volatility by strike, expiry {mkt['expiry']} "
                 f"(quotes of {mkt['quote_date']})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "q5_iv_smile.png", dpi=150)
    plt.close(fig)
    return float(atm_row.iv), float(atm_row.strike)


def q6(mkt, atm_iv):
    rv, n = realised_vol(mkt["closes"])
    print(f"\nQ6  {TICKER} volatility, realised vs implied")
    print(f"    realised, last {n} daily returns to {mkt['quote_date']}, annualised: {rv:.2%}")
    print(f"    implied, at the money, {mkt['expiry']}:                        {atm_iv:.2%}")
    print(f"    implied minus realised: {atm_iv - rv:+.2%}")
    return rv


# ---------------------------------------------------------------- Q7
def load_history(snapshot=False):
    """Daily SPY and VIX closes since SIM_START, saved for snapshot reruns."""
    path = DATA / "spy_vix_history.csv"
    if not snapshot:
        spy = yf.download(TICKER, start=SIM_START, auto_adjust=False, progress=False)["Close"].squeeze()
        vix = yf.download("^VIX", start=SIM_START, auto_adjust=False, progress=False)["Close"].squeeze()
        pd.DataFrame({"spy": spy, "vix": vix}).dropna().to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def vix_vs_realised(hist, horizon=21):
    """How often the VIX exceeded the volatility SPY then realised over the next month."""
    returns = np.log(hist["spy"] / hist["spy"].shift(1))
    future_rv = returns[::-1].rolling(horizon).std()[::-1].shift(-1) * sqrt(TRADING_DAYS) * 100
    both = pd.DataFrame({"vix": hist["vix"], "rv": future_rv}).dropna()
    return (both["vix"] > both["rv"]).mean(), (both["vix"] - both["rv"]).mean(), len(both)


def leveraged_account(pnl, leverage):
    """Compound weekly P/L on an account that sells `leverage` dollars of notional per dollar.

    Returns (final multiple, worst drawdown, date the account hit zero or None).
    """
    equity, peak, worst = 1.0, 1.0, 0.0
    for d, r in pnl.items():
        equity *= 1 + leverage * r
        if equity <= 0:
            return 0.0, -1.0, d
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1)
    return equity, worst, None


def short_straddle_weekly(hist, rate=0.0, hold=5, vol_shift=0.0):
    """Sell a 1-week at-the-money straddle every week, hold to expiry, repeat.

    Entry every `hold` trading days at the close. Both legs are priced with
    Black-Scholes at that day's VIX as the volatility, struck at the close, and
    settled at the close `hold` trading days later. Profit is per $1 of SPY
    sold against, so weeks in 2005 and 2026 are comparable. No costs, no
    margin calls, no early exercise: the most flattering version of the trade.

    The VIX includes the put skew, so it sits above at-the-money implied vol
    (19.4 vs 16.0 on the run date). `vol_shift` (in vol points) prices the
    straddle below the VIX to see how much of the edge that alone accounts for.
    """
    rows = []
    idx = hist.index
    for i in range(0, len(hist) - hold, hold):
        s0, s1 = hist["spy"].iloc[i], hist["spy"].iloc[i + hold]
        vol = max(hist["vix"].iloc[i] + vol_shift, 1.0) / 100
        years = (idx[i + hold] - idx[i]).days / 365
        premium = (black_scholes("call", s0, s0, years, rate, vol)
                   + black_scholes("put", s0, s0, years, rate, vol))
        rows.append((idx[i], (premium - abs(s1 - s0)) / s0))
    return pd.Series(dict(rows), name="pnl")


def q7(mkt, atm_iv, rv, snapshot=False):
    print(f"\nQ7  is implied above realised?")
    gap = atm_iv - rv
    print(f"    today: implied {atm_iv:.2%} vs realised {rv:.2%} -> "
          f"{'YES, implied is higher' if gap > 0 else 'NO, realised is higher'} by {abs(gap):.2%}")
    hist = load_history(snapshot)
    frac, mean_gap, n = vix_vs_realised(hist)
    print(f"    since {hist.index[0].date()}: the VIX was above the next month's realised SPY vol on "
          f"{frac:.0%} of {n} days, by {mean_gap:+.1f} vol points on average")

    pnl = short_straddle_weekly(hist)
    cum = pnl.cumsum()
    wins = (pnl > 0).mean()
    print(f"\n    selling a 1-week ATM SPY straddle every week since {pnl.index[0].date()} "
          f"({len(pnl)} weeks, priced at the VIX, no costs):")
    print(f"    profitable weeks {wins:.0%}   average week {pnl.mean():+.3%} of notional   "
          f"median week {pnl.median():+.3%}")
    print(f"    average winning week {pnl[pnl > 0].mean():+.3%}   "
          f"average losing week {pnl[pnl <= 0].mean():+.3%}")
    print("    five worst weeks (loss as % of notional, against the median week's gain):")
    for d, v in pnl.nsmallest(5).items():
        print(f"      {d.date()}  {v:+.2%}   = {v / pnl.median():+.0f} median weeks")
    peak_dd = (cum - cum.cummax()).min()
    print(f"    cumulative {cum.iloc[-1]:+.1%} of notional; worst peak-to-trough {peak_dd:+.1%}")
    for start, end, label in [("2008-09-01", "2008-12-31", "autumn 2008"),
                              ("2020-02-14", "2020-04-30", "Feb-Apr 2020")]:
        window = pnl[start:end]
        if len(window):
            print(f"    {label}: {window.sum():+.1%} over {len(window)} weeks")

    cheap = short_straddle_weekly(hist, vol_shift=-3.0)
    print(f"\n    the same trade priced 3 vol points below the VIX (closer to today's ATM gap):")
    print(f"    profitable weeks {(cheap > 0).mean():.0%}   average week {cheap.mean():+.3%}   "
          f"cumulative {cheap.sum():+.1%}")

    # Priced at the full VIX the edge is too generous to compound sensibly
    # (an unlevered +25% a year on notional), so the account uses VIX - 3.
    print("\n    on a margin account, priced at VIX - 3, compounding, selling L dollars of "
          "notional per dollar of equity:")
    for lev in (1, 3, 5, 8):
        final, dd, ruin = leveraged_account(cheap, lev)
        result = (f"wiped out in the week of {ruin.date()}" if ruin is not None
                  else f"ends at x{final:,.1f} the starting equity, worst drawdown {dd:+.0%}")
        print(f"    L = {lev}: {result}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    ax.plot(cum.index, cum * 100, lw=1.4, color="#2b6cb0", label="priced at the VIX")
    ax.plot(cheap.index, cheap.cumsum() * 100, lw=1.4, color="#c05621", label="priced at VIX - 3")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("cumulative P/L, % of notional (summed)")
    ax.set_title("Q7: selling a 1-week ATM SPY straddle every week\n"
                 f"profitable {wins:.0%} of weeks; the losses arrive together")
    ax.grid(alpha=0.3)
    ax.legend()
    ax = axes[1]
    ax.bar(pnl.index, pnl * 100, width=5, color=np.where(pnl > 0, "#2f855a", "#c53030"))
    ax.set_ylabel("weekly P/L, % of notional")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "q7_short_straddle.png", dpi=150)
    plt.close(fig)


def main(snapshot=False):
    mkt = load_market(snapshot)
    print(f"run date {mkt['run_date']}; {TICKER} quotes from the {mkt['quote_date']} close\n")
    atm = q1(mkt)
    atm_iv, atm_strike = q5_quiet(mkt)
    q2(mkt, atm_iv, atm)
    q3()
    q4()
    atm_iv, _ = q5(mkt)
    rv = q6(mkt, atm_iv)
    q7(mkt, atm_iv, rv, snapshot)


def q5_quiet(mkt):
    """The ATM implied vol, computed without printing, for the Q2 illustration."""
    calls = smile(clean_quotes(mkt["calls"], mkt["quote_date"]), mkt, "call").dropna(subset=["iv"])
    row = calls.loc[(calls["strike"] - mkt["spot"]).abs().idxmin()]
    return float(row.iv), float(row.strike)


if __name__ == "__main__":
    main(snapshot="--snapshot" in sys.argv)
