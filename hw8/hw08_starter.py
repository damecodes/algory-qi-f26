"""Homework 8 starter — Algory QI Education, Fall 2026

Price an option from history alone, then compare it with Black-Scholes.

    python hw08_starter.py              # download 20 years of SPY, save to data/, run Q1-Q7
    python hw08_starter.py --snapshot   # rerun on the saved data/ files

Every live run saves what it used to data/, and both modes then run on those
saved files, so the numbers in the write-up can be reproduced later.
"""
import importlib.util
import json
import pathlib
import sys
from datetime import date
from math import exp, sqrt

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
HW6_FILE = HERE.parent / "hw6" / "hw06_starter.py"

TICKER = "SPY"
YEARS = 20
SPOT = 760.88              # the current spot given in the assignment
DAYS = 21                  # one month of trading days
CALL_OTM = 0.03            # Q3: call struck 3% above spot
RV_WINDOW = 60             # Q5: realised volatility over the past 60 days
TRADING_DAYS = 252
STRIKE_GRID = np.round(np.arange(0.85, 1.1501, 0.01), 2)   # Q6: 15% below to 15% above

# plot colours: historical method in blue, Black-Scholes / normal in orange
HIST_C = "#2a78d6"
BS_C = "#eb6834"
MUTED = "#8a8985"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "axes.edgecolor": "#8a8985", "lines.linewidth": 2,
})


# ---------------------------------------------------------------- the two functions
def n_day_returns(prices, n):
    """Every overlapping n-day return in the series.

    A series of length L gives about L - n windows.
    With n = 1 this is just the daily return series.
    """
    p = np.asarray(prices, dtype=float).ravel()
    if n < 1:
        raise ValueError("n must be at least 1")
    if len(p) <= n:
        return np.array([])
    # window i starts at close i and ends at close i + n
    return p[n:] / p[:-n] - 1.0


def payoff(kind, strike, final):
    """Payoff at expiry of one call or put, for an array of final prices."""
    if kind == "call":
        return np.maximum(final - strike, 0.0)
    if kind == "put":
        return np.maximum(strike - final, 0.0)
    raise ValueError(f"kind must be 'call' or 'put', not {kind!r}")


def final_prices(prices, spot, days):
    """Today's spot moved by every historical `days`-day return."""
    return spot * (1.0 + n_day_returns(prices, days))


def price_from_history(prices, spot, strike, days, kind):
    """Price an option from what the underlying has actually done.

    Apply every historical `days`-day return to `spot` to build a distribution of
    final prices, compute the payoff under each, and return the mean. No
    distributional assumption anywhere.

    Sanity checks: a strike far below spot prices near intrinsic; a strike far
    above prices near zero; nothing is ever negative.

    Not discounted: this is the plain average payoff, with no risk-free rate.
    """
    return float(np.mean(payoff(kind, strike, final_prices(prices, spot, days))))


# ---------------------------------------------------------------- data
def download_spy(years=YEARS):
    """Daily SPY closes for the last `years` years.

    The option is written on SPY's price, and the price drops by the dividend on
    each ex-date, so the raw close (auto_adjust=False) is the right series: a
    dividend-adjusted close would build in about 1.1% a year of dividends that a
    call holder never receives. The adjusted close is saved for reference only.
    """
    end = pd.Timestamp(date.today())
    start = end - pd.DateOffset(years=years)
    raw = yf.download(TICKER, start=start.strftime("%Y-%m-%d"),
                      end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                      auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(TICKER, axis=1, level="Ticker")
    df = raw[["Close", "Adj Close"]].rename(columns={"Close": "close",
                                                     "Adj Close": "adj_close"}).dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df


def download_tbill():
    """Latest 13-week T-bill yield (^IRX), in percent a year. Used only for the discounting note."""
    raw = yf.download("^IRX", period="1mo", auto_adjust=True, progress=False)["Close"]
    raw = raw.squeeze().dropna()
    return float(raw.iloc[-1]), str(pd.Timestamp(raw.index[-1]).date())


def save_snapshot(spy, irx, irx_date):
    DATA.mkdir(exist_ok=True)
    spy.to_csv(DATA / "SPY_daily.csv")
    meta = {"run_date": str(date.today()), "ticker": TICKER, "years": YEARS,
            "first": str(spy.index[0].date()), "last": str(spy.index[-1].date()),
            "rows": len(spy), "last_close": round(float(spy["close"].iloc[-1]), 4),
            "spot_used": SPOT,
            "source": "yfinance download, auto_adjust=False (raw close)",
            "tbill_pct": irx, "tbill_date": irx_date}
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def load_snapshot():
    spy = pd.read_csv(DATA / "SPY_daily.csv", index_col="date", parse_dates=True)
    meta = json.loads((DATA / "meta.json").read_text())
    return spy, meta


def load_hw6():
    """Import the Homework 6 module so its black_scholes() and implied_volatility() are used as written."""
    spec = importlib.util.spec_from_file_location("hw06_starter", HW6_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def realised_vol(closes, window=RV_WINDOW):
    """Annualised standard deviation of the last `window` daily log returns (same as HW6)."""
    returns = np.log(closes / closes.shift(1)).dropna().iloc[-window:]
    return float(returns.std(ddof=1) * sqrt(TRADING_DAYS)), returns.index[0], returns.index[-1]


def episodes(closes, mask, days=DAYS):
    """Group the windows flagged in `mask` into distinct market episodes.

    Overlapping windows that start within `days` of each other share most of
    their path, so they are one event, not several. Returns (first start, last end)
    date pairs.
    """
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > days)
    groups = np.split(idx, breaks + 1)
    dates = closes.index
    return [(dates[g[0]].date(), dates[g[-1] + days].date()) for g in groups]


def pct(x, digits=2):
    return f"{100 * x:+.{digits}f}%"


def header(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


# ---------------------------------------------------------------- Q1
def q1(closes):
    header("Q1. The 21-day return distribution (overlapping windows)")
    r = n_day_returns(closes.values, DAYS)
    starts = closes.index[:-DAYS]
    lo, hi = int(np.argmin(r)), int(np.argmax(r))
    print(f"Trading days of SPY closes      : {len(closes):,} "
          f"({closes.index[0].date()} to {closes.index[-1].date()})")
    print(f"Overlapping {DAYS}-day windows       : {len(r):,}")
    print(f"Mean {DAYS}-day return              : {pct(r.mean(), 3)}")
    print(f"Standard deviation              : {100 * r.std(ddof=1):.3f}%")
    print(f"1st percentile                  : {pct(np.percentile(r, 1))}")
    print(f"99th percentile                 : {pct(np.percentile(r, 99))}")
    print(f"Median                          : {pct(np.median(r), 3)}")
    print(f"Worst window                    : {pct(r[lo])}, starting {starts[lo].date()}")
    print(f"Best window                     : {pct(r[hi])}, starting {starts[hi].date()}")
    return r


# ---------------------------------------------------------------- Q2
def q2(closes, r):
    header("Q2. Shape of the distribution against a normal")
    mu, sd = r.mean(), r.std(ddof=1)
    skew = stats.skew(r, bias=False)
    exkurt = stats.kurtosis(r, fisher=True, bias=False)
    daily = n_day_returns(closes.values, 1)
    d_skew = stats.skew(daily, bias=False)
    d_kurt = stats.kurtosis(daily, fisher=True, bias=False)
    print(f"{DAYS}-day returns : skewness {skew:+.3f}, excess kurtosis {exkurt:+.3f}")
    print(f"Daily returns  : skewness {d_skew:+.3f}, excess kurtosis {d_kurt:+.3f}")
    for k in (3, 4):
        below = int(np.sum(r < mu - k * sd))
        above = int(np.sum(r > mu + k * sd))
        expect = len(r) * stats.norm.cdf(-k)
        print(f"Windows beyond {k} sd: {below} below and {above} above the mean; "
              f"a normal expects {expect:.1f} on each side")

    x = np.linspace(r.min() - 0.01, r.max() + 0.01, 600)
    bins = np.linspace(r.min(), r.max(), 90)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, log in zip(axes, (False, True)):
        ax.hist(r, bins=bins, density=True, color=HIST_C, alpha=0.75,
                edgecolor="white", linewidth=0.4, label=f"SPY {DAYS}-day returns")
        ax.plot(x, stats.norm.pdf(x, mu, sd), color=BS_C, label="normal, same mean and sd")
        ax.set_xlabel(f"{DAYS}-day return")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e-2, None)
            ax.set_title("Same, log density scale: the tails")
        else:
            ax.set_title(f"{len(r):,} overlapping {DAYS}-day SPY returns, "
                         f"{closes.index[0].year} to {closes.index[-1].year}")
            ax.set_ylabel("density")
            ax.text(0.02, 0.95, f"skew {skew:+.2f}\nexcess kurtosis {exkurt:+.2f}",
                    transform=ax.transAxes, va="top", fontsize=9)
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / "q2_return_distribution.png", dpi=150)
    plt.close(fig)
    print("Plot: q2_return_distribution.png")
    return skew, exkurt


# ---------------------------------------------------------------- Q3
def q3(closes, rate):
    header("Q3. Pricing from history (undiscounted)")
    strike = SPOT * (1 + CALL_OTM)
    call = price_from_history(closes.values, SPOT, strike, DAYS, "call")
    put = price_from_history(closes.values, SPOT, strike, DAYS, "put")
    mean_final = float(np.mean(final_prices(closes.values, SPOT, DAYS)))
    print(f"Spot ${SPOT:.2f}, strike ${strike:.2f} (3% above spot), {DAYS} trading days")
    print(f"Historical call price           : ${call:.4f}")
    print(f"Historical put price            : ${put:.4f}")
    print(f"Call minus put                  : ${call - put:+.4f}")
    print(f"Spot minus strike               : ${SPOT - strike:+.4f}")
    print(f"Mean final price minus strike   : ${mean_final - strike:+.4f}  "
          f"(exact identity: the mean final price is ${mean_final:.2f})")
    disc = exp(-rate * DAYS / TRADING_DAYS)
    print(f"If discounted at the {100 * rate:.3f}% T-bill rate for {DAYS}/{TRADING_DAYS} of a year "
          f"(factor {disc:.5f}): call ${call * disc:.4f}, put ${put * disc:.4f}")
    return strike, call, put


# ---------------------------------------------------------------- Q4
def q4(closes, strike):
    header("Q4. Distribution of final prices")
    final = final_prices(closes.values, SPOT, DAYS)
    itm_call = float(np.mean(final > strike))
    itm_put = float(np.mean(final < strike))
    print(f"Final prices: {len(final):,}, from ${final.min():.2f} to ${final.max():.2f}")
    print(f"Call finishes in the money in {itm_call:.2%} of windows "
          f"({int(np.sum(final > strike)):,} of {len(final):,})")
    print(f"Put finishes in the money in  {itm_put:.2%} of windows "
          f"({int(np.sum(final < strike)):,} of {len(final):,})")
    print(f"Average call payoff when it pays: ${np.mean(final[final > strike] - strike):.2f}")
    print(f"Average put payoff when it pays : ${np.mean(strike - final[final < strike]):.2f}")

    bins = np.linspace(final.min(), final.max(), 100)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, kind, frac in zip(axes, ("call", "put"), (itm_call, itm_put)):
        pays = final > strike if kind == "call" else final < strike
        ax.hist(final[~pays], bins=bins, color="#c9c8c2", edgecolor="white",
                linewidth=0.4, label="expires worthless")
        ax.hist(final[pays], bins=bins, color=HIST_C, edgecolor="white",
                linewidth=0.4, label=f"pays off ({frac:.1%} of windows)")
        ax.axvline(strike, color="#0b0b0b", linewidth=1.2)
        ax.axvline(SPOT, color=MUTED, linewidth=1, linestyle="--")
        ymax = ax.get_ylim()[1]
        ax.text(strike, ymax * 0.97, f" strike ${strike:.2f}", va="top", fontsize=8)
        ax.text(SPOT, ymax * 0.88, f"spot ${SPOT:.2f} ", va="top", ha="right",
                fontsize=8, color="#52514e")
        ax.set_title(f"{DAYS}-day {kind} struck 3% above spot")
        ax.set_xlabel("SPY price at expiry ($)")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.8))
    axes[0].set_ylabel("number of historical windows")
    fig.suptitle(f"Spot ${SPOT:.2f} moved by each of the {len(final):,} historical "
                 f"{DAYS}-day returns", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "q4_final_prices.png", dpi=150)
    plt.close(fig)
    print("Plot: q4_final_prices.png")
    return itm_call, itm_put


# ---------------------------------------------------------------- Q5
def q5(closes, hw6, strike, call, put, rate):
    header("Q5. Black-Scholes with 60-day realised volatility")
    rv, rv_from, rv_to = realised_vol(closes, RV_WINDOW)
    years = DAYS / TRADING_DAYS
    hist_vol = float(np.std(np.log1p(n_day_returns(closes.values, DAYS)), ddof=1)
                     * sqrt(TRADING_DAYS / DAYS))
    print(f"Realised vol, last {RV_WINDOW} daily log returns ({rv_from.date()} to {rv_to.date()}): "
          f"{100 * rv:.2f}% a year")
    print(f"For reference, the 20-year sample's own vol ({DAYS}-day log returns, annualised): "
          f"{100 * hist_vol:.2f}% a year")
    print(f"Time {DAYS}/{TRADING_DAYS} = {years:.4f} years. Black-Scholes at rate 0 is the like-for-like")
    print("comparison: the historical prices are undiscounted, so the main BS price is too.\n")
    print(f"{'contract':<20}{'historical':>12}{'BS r=0':>12}{'hist vs BS':>12}"
          f"{'BS r=T-bill':>14}{'BS at 20y vol':>15}")
    out = {}
    for kind, hist in (("call", call), ("put", put)):
        bs0 = hw6.black_scholes(kind, SPOT, strike, years, 0.0, rv)
        bsr = hw6.black_scholes(kind, SPOT, strike, years, rate, rv)
        bsh = hw6.black_scholes(kind, SPOT, strike, years, 0.0, hist_vol)
        out[kind] = bs0
        print(f"{kind + f' K={strike:.2f}':<20}{hist:>12.4f}{bs0:>12.4f}"
              f"{pct(hist / bs0 - 1, 1):>12}{bsr:>14.4f}{bsh:>15.4f}")
    print(f"\n(hist vs BS = historical / Black-Scholes - 1. The T-bill rate is {100 * rate:.3f}%.)")
    return rv, hist_vol, out


# ---------------------------------------------------------------- Q6
def q6(closes, hw6, rv, hist_vol):
    header("Q6. Across strikes, 15% below spot to 15% above")
    years = DAYS / TRADING_DAYS
    prices = closes.values
    # The historical forward is the mean final price, which carries SPY's drift.
    # To read an implied vol off a historical price, invert Black-Scholes at the
    # rate that reproduces that forward, and discount the price at the same rate
    # (BS returns e^-rT times the expected payoff). Then puts and calls on either
    # side of spot share one forward, as they do inside the historical method.
    fwd = float(np.mean(final_prices(prices, SPOT, DAYS)))
    drift = np.log(fwd / SPOT) / years
    rows = []
    for m in STRIKE_GRID:
        k = SPOT * m
        row = {"moneyness": m, "strike": k}
        for kind in ("call", "put"):
            row[f"hist_{kind}"] = price_from_history(prices, SPOT, k, DAYS, kind)
            row[f"bs_{kind}"] = hw6.black_scholes(kind, SPOT, k, years, 0.0, rv)
        otm = "put" if m < 1 else "call"
        row["otm"] = otm
        row["hist_iv"] = hw6.implied_volatility(row[f"hist_{otm}"] * exp(-drift * years),
                                                SPOT, k, years, drift, otm)
        rows.append(row)
    df = pd.DataFrame(rows)
    for kind in ("call", "put"):
        df[f"diff_{kind}"] = df[f"hist_{kind}"] - df[f"bs_{kind}"]
    df["hist_otm"] = np.where(df.otm == "put", df.hist_put, df.hist_call)
    df["bs_otm"] = np.where(df.otm == "put", df.bs_put, df.bs_call)
    df["ratio_otm"] = df.hist_otm / df.bs_otm

    print(f"Black-Scholes at rate 0 and {100 * rv:.2f}% vol. Prices in dollars.\n")
    print(f"{'strike':>8}{'K/S':>6} | {'hist call':>10}{'BS call':>10}{'diff':>9} | "
          f"{'hist put':>10}{'BS put':>10}{'diff':>9} | {'OTM':>4}{'hist/BS':>10}{'hist IV':>9}")
    for _, r in df.iterrows():
        ratio = f"{r.ratio_otm:>10.2f}" if r.ratio_otm < 1e4 else f"{r.ratio_otm:>10.2e}"
        print(f"{r.strike:>8.2f}{r.moneyness:>6.2f} | {r.hist_call:>10.4f}{r.bs_call:>10.4f}"
              f"{r.diff_call:>+9.3f} | {r.hist_put:>10.4f}{r.bs_put:>10.4f}{r.diff_put:>+9.3f} | "
              f"{r.otm:>4}{ratio}{100 * r.hist_iv:>8.2f}%")

    print(f"\nHistorical forward (mean final price) ${fwd:.2f} vs Black-Scholes forward at r=0 "
          f"${SPOT:.2f}: a ${fwd - SPOT:+.2f} gap. By put-call parity inside each method,")
    print("every call's dollar difference equals the same-strike put's difference plus that gap.")
    long = pd.concat([df.assign(kind=k, diff=df[f"diff_{k}"])[["strike", "moneyness", "kind", "diff"]]
                      for k in ("call", "put")], ignore_index=True)
    worst = long.loc[long["diff"].abs().idxmax()]
    print(f"Largest disagreement in dollars, any option: the {worst.kind} at ${worst.strike:.2f} "
          f"({worst.moneyness:.2f} x spot), historical minus BS = ${worst['diff']:+.3f}")
    df["diff_otm"] = df.hist_otm - df.bs_otm
    w = df.loc[df.diff_otm.abs().idxmax()]
    print(f"Largest disagreement in dollars, out-of-the-money options only: the {w.otm} at "
          f"${w.strike:.2f} ({w.moneyness:.2f} x spot), historical ${w.hist_otm:.4f} vs BS "
          f"${w.bs_otm:.4f}, difference ${w.diff_otm:+.3f}")
    rel = df.loc[df.ratio_otm.idxmax()]
    print(f"Largest disagreement in relative terms (out-of-the-money options): the {rel.otm} "
          f"at ${rel.strike:.2f} ({rel.moneyness:.2f} x spot), historical ${rel.hist_otm:.4f} vs "
          f"BS ${rel.bs_otm:.6f}, {rel.ratio_otm:,.0f} times as much")
    otm_puts = df[df.moneyness <= 0.90]
    print("Deep out-of-the-money puts (strike 10% or more below spot): historical price is "
          f"{otm_puts.ratio_otm.min():,.0f} to {otm_puts.ratio_otm.max():,.0f} times Black-Scholes")
    print(f"Historical-implied vol (inverted at the historical forward): {100 * df.hist_iv.iloc[0]:.1f}% at 0.85 x spot, "
          f"{100 * df.loc[df.moneyness == 1.0, 'hist_iv'].iloc[0]:.1f}% at the money, "
          f"{100 * df.hist_iv.iloc[-1]:.1f}% at 1.15 x spot; "
          f"minimum {100 * df.hist_iv.min():.1f}% at {df.loc[df.hist_iv.idxmin(), 'moneyness']:.2f} x spot")

    # plots: put prices, call prices, dollar difference, implied vol of the historical prices
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    k = df.strike
    for ax, kind in zip(axes[0], ("put", "call")):
        ax.plot(k, df[f"hist_{kind}"], color=HIST_C, label="historical")
        ax.plot(k, df[f"bs_{kind}"], color=BS_C, label=f"Black-Scholes, {100 * rv:.1f}% vol")
        ax.axvline(SPOT, color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(f"{DAYS}-day {kind} prices")
        ax.set_xlabel("strike ($)")
        ax.set_ylabel("price ($)")
        ax.legend(frameon=False)
    ax = axes[1, 0]
    ax.plot(k, df.diff_put, color="#6250d6", label="puts")
    ax.plot(k, df.diff_call, color="#1baf7a", label="calls")
    ax.axhline(0, color="#0b0b0b", linewidth=0.8)
    ax.axvline(SPOT, color=MUTED, linewidth=1, linestyle="--")
    ax.set_title("Historical minus Black-Scholes")
    ax.set_xlabel("strike ($)")
    ax.set_ylabel("difference ($)")
    ax.legend(frameon=False)
    ax = axes[1, 1]
    ax.plot(k, 100 * df.hist_iv, color=HIST_C, marker="o", markersize=4,
            label="vol implied by the historical price")
    ax.axhline(100 * rv, color=BS_C, label=f"Black-Scholes input ({RV_WINDOW}-day realised)")
    ax.axhline(100 * hist_vol, color=MUTED, linestyle=":", linewidth=1.5,
               label="20-year sample vol")
    ax.axvline(SPOT, color=MUTED, linewidth=1, linestyle="--")
    ax.set_title("Implied vol of the historical prices (OTM options)")
    ax.text(0.98, 0.95, "historical drift built into the forward", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color="#52514e")
    ax.set_xlabel("strike ($)")
    ax.set_ylabel("implied volatility (%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "q6_strike_comparison.png", dpi=150)
    plt.close(fig)
    print("Plot: q6_strike_comparison.png")
    return df


# ---------------------------------------------------------------- Q7 support
def q7_tails(closes, r, hw6, rv):
    header("Q7 support. How much data is really in the tails")
    years = DAYS / TRADING_DAYS
    print(f"{len(r):,} overlapping windows, but only about {len(r) // DAYS} non-overlapping "
          f"{DAYS}-day periods in the sample.")
    for thr in (-0.10, -0.15, -0.20, 0.10, 0.15, 0.20):
        mask = r < thr if thr < 0 else r > thr
        eps = episodes(closes, mask)
        span = "; ".join(f"{a} to {b}" for a, b in eps) if eps else "none"
        side = "below" if thr < 0 else "above"
        print(f"\nReturns {side} {pct(thr, 0)}: {int(mask.sum())} windows "
              f"({mask.mean():.2%}), {len(eps)} distinct episode(s)")
        print(f"   {span}")

    print("\nThe same price computed on each of the 21 disjoint subsamples "
          "(windows starting every 21st day):")
    print(f"{'contract':<24}{'all windows':>12}{'min':>10}{'median':>10}{'max':>10}{'BS':>12}")
    for kind, m in (("call", 1.03), ("put", 1.03), ("put", 0.90), ("put", 0.85),
                    ("put", 0.80), ("call", 1.20)):
        k = SPOT * m
        full = price_from_history(closes.values, SPOT, k, DAYS, kind)
        subs = []
        for off in range(DAYS):
            pay = payoff(kind, k, SPOT * (1 + r[off::DAYS]))
            subs.append(pay.mean())
        bs = hw6.black_scholes(kind, SPOT, k, years, 0.0, rv)
        print(f"{f'{kind} at {m:.2f} x spot':<24}{full:>12.4f}{min(subs):>10.4f}"
              f"{np.median(subs):>10.4f}{max(subs):>10.4f}{bs:>12.6f}")


# ---------------------------------------------------------------- main
def main(snapshot=False):
    if not snapshot:
        spy = download_spy()
        irx, irx_date = download_tbill()
        save_snapshot(spy, irx, irx_date)
    # Always run on the saved files, so a live run and a --snapshot run see the same data.
    spy, meta = load_snapshot()
    closes = spy["close"]
    rate = meta["tbill_pct"] / 100
    print(f"SPY raw closes {meta['first']} to {meta['last']}, {meta['rows']:,} rows "
          f"({'saved snapshot' if snapshot else 'live download'}, run {meta['run_date']}).")
    print(f"Last close in the data ${meta['last_close']:.2f}; the assignment's spot "
          f"${SPOT:.2f} is used throughout. 13-week T-bill {meta['tbill_pct']:.3f}% "
          f"on {meta['tbill_date']}.")
    hw6 = load_hw6()

    r = q1(closes)
    q2(closes, r)
    strike, call, put = q3(closes, rate)
    q4(closes, strike)
    rv, hist_vol, _ = q5(closes, hw6, strike, call, put, rate)
    q6(closes, hw6, rv, hist_vol)
    q7_tails(closes, r, hw6, rv)


if __name__ == "__main__":
    main(snapshot="--snapshot" in sys.argv)
