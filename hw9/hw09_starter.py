"""Homework 9 starter — Algory QI Education, Fall 2026

Volatility, tails and value at risk on 20 years of daily SPY.

    python hw09_starter.py              # download 20 years of SPY, save to data/, run Q1-Q9
    python hw09_starter.py --snapshot   # rerun on the saved data/ files

Every live run saves what it used to data/, and both modes then run on those
saved files, so the numbers in the write-up can be reproduced later.

The five functions below are what check_hw09.py tests. simulate_paths and
bootstrap_paths are written to be reused for position sizing in Session 10.
"""
import json
import pathlib
import sys
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"

TICKER = "SPY"
YEARS = 20
TRADING_DAYS = 252
ROLL_WINDOW = 60           # Q3: rolling volatility window, in trading days
SIGMA_CUT = 4.0            # Q4/Q5: what counts as a tail day
LOSSES = [0.10, 0.25, 0.50, 0.75, 0.90]   # Q6
START_VALUE = 100_000.0    # Q7
N_PATHS = 10_000           # Q7
CONFIDENCE = 0.95          # Q8
BLOCK = 21                 # Q9 extra: block length for the block bootstrap
SEED = 42

# Spike labels, looked up from the date of each rolling-volatility peak
EPISODES = {
    (2008, 9): "2008 financial crisis", (2008, 10): "2008 financial crisis",
    (2008, 11): "2008 financial crisis", (2008, 12): "2008 financial crisis",
    (2009, 1): "2008 financial crisis", (2009, 3): "2009 bear-market low",
    (2010, 5): "2010 flash crash / euro debt", (2010, 6): "2010 flash crash / euro debt",
    (2011, 8): "2011 US downgrade / euro crisis", (2011, 9): "2011 US downgrade / euro crisis",
    (2011, 10): "2011 US downgrade / euro crisis",
    (2015, 8): "2015 China devaluation", (2015, 9): "2015 China devaluation",
    (2018, 2): "Feb 2018 volatility spike", (2018, 12): "Q4 2018 selloff",
    (2019, 1): "Q4 2018 selloff",
    (2020, 3): "2020 COVID crash", (2020, 4): "2020 COVID crash", (2020, 5): "2020 COVID crash",
    (2022, 6): "2022 rate shock", (2022, 7): "2022 rate shock", (2022, 10): "2022 rate shock",
    (2025, 4): "2025 tariff shock", (2025, 5): "2025 tariff shock",
}

# plot colours: data in blue, normal / GBM in orange, bootstrap in green
DATA_C = "#2a78d6"
NORM_C = "#eb6834"
BOOT_C = "#2f9e6e"
MUTED = "#8a8985"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "axes.edgecolor": "#8a8985", "lines.linewidth": 2,
})


# ---------------------------------------------------------------- the five functions
def recovery_needed(loss_fraction):
    """Gain required to get back to even after a loss.

    recovery_needed(0.50) -> 1.00     lose half, need to double
    recovery_needed(0.75) -> 3.00     lose three quarters, need +300%

    After a loss L you hold (1 - L) of what you had; growing back to 1 needs a
    gain g with (1 - L)(1 + g) = 1, so g = L / (1 - L).
    """
    loss = np.asarray(loss_fraction, dtype=float)
    if np.any((loss < 0) | (loss >= 1)):
        raise ValueError("loss_fraction must be in [0, 1)")
    g = loss / (1.0 - loss)
    return float(g) if g.ndim == 0 else g


def annualise_vol(daily_vol, days=TRADING_DAYS):
    """Scale a daily standard deviation to annual. Assumes days are independent."""
    return daily_vol * np.sqrt(days)


def var(returns, confidence=CONFIDENCE):
    """Value at risk, as a return. A left-tail quantile, so it comes out negative.

    At 95% this is the 5th percentile of the returns: the outcome that is beaten
    95% of the time. The loss in money is -var * portfolio value.
    """
    r = np.asarray(returns, dtype=float).ravel()
    return float(np.quantile(r, 1.0 - confidence))


def cvar(returns, confidence=CONFIDENCE):
    """Mean of everything at or beyond the VaR threshold.

    Must always be worse than var() on the same inputs. If it is not, you have
    averaged the wrong side of the distribution.
    """
    r = np.asarray(returns, dtype=float).ravel()
    return float(r[r <= var(r, confidence)].mean())


def simulate_paths(start, drift, vol, days=TRADING_DAYS, paths=1000, seed=0):
    """Geometric Brownian motion. Return the FINAL value of each path.

    `drift` and `vol` are annual. `drift` is the arithmetic drift mu, so the
    expected final value is start * exp(mu * T). Each day's log return is
    normal with mean (mu - vol^2 / 2) dt and standard deviation vol * sqrt(dt).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS
    z = rng.standard_normal((paths, days))
    log_paths = ((drift - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * z).sum(axis=1)
    return start * np.exp(log_paths)


# ---------------------------------------------------------------- bootstrap (Q9)
def bootstrap_paths(start, daily_returns, days=TRADING_DAYS, paths=1000, seed=0, block=1):
    """Final values from resampling actual daily returns, with replacement.

    block=1 draws every day independently, which keeps the fat-tailed daily
    distribution but throws away volatility clustering. block > 1 draws runs of
    `block` consecutive days, which keeps the clustering inside each run.
    """
    r = np.asarray(daily_returns, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    log_r = np.log1p(r)
    if block == 1:
        idx = rng.integers(0, len(r), size=(paths, days))
    else:
        n_blocks = -(-days // block)
        starts = rng.integers(0, len(r) - block + 1, size=(paths, n_blocks))
        idx = (starts[:, :, None] + np.arange(block)).reshape(paths, -1)[:, :days]
    return start * np.exp(log_r[idx].sum(axis=1))


# ---------------------------------------------------------------- data
def download_spy(years=YEARS):
    """Daily SPY closes for the last `years` years.

    Risk here is measured on a holder's total return, so the analysis uses the
    dividend-adjusted close. The raw close is saved alongside for reference.
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


def load_data(snapshot):
    csv = DATA / "SPY_daily.csv"
    if not snapshot:
        DATA.mkdir(exist_ok=True)
        df = download_spy()
        df.to_csv(csv)
        meta = {"run_date": date.today().isoformat(), "ticker": TICKER, "years": YEARS,
                "first": df.index[0].date().isoformat(), "last": df.index[-1].date().isoformat(),
                "rows": len(df), "source": "yfinance download, auto_adjust=False; "
                "analysis uses adj_close (total return)"}
        (DATA / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not csv.exists():
        sys.exit("No saved data: run once without --snapshot first.")
    df = pd.read_csv(csv, index_col="date", parse_dates=True)
    return df["adj_close"]


# ---------------------------------------------------------------- helpers
def spike_episodes(roll, n=5, gap_days=TRADING_DAYS):
    """The n highest rolling-vol peaks, at least `gap_days` trading days apart."""
    s = roll.dropna().copy()
    peaks = []
    while len(peaks) < n and s.notna().any():
        d = s.idxmax()
        peaks.append((d, float(s[d])))
        i = roll.index.get_loc(d)
        lo, hi = roll.index[max(i - gap_days, 0)], roll.index[min(i + gap_days, len(roll) - 1)]
        s[lo:hi] = np.nan
    return peaks


def episode_name(d):
    return EPISODES.get((d.year, d.month), f"{d:%B %Y}")


def money(x):
    return f"${x:,.0f}"


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}%"


# ---------------------------------------------------------------- main
def main():
    snapshot = "--snapshot" in sys.argv
    px = load_data(snapshot)
    r = px.pct_change().dropna()
    lr = np.log(px).diff().dropna()
    n = len(r)
    print(f"Data: {TICKER} adjusted close, {px.index[0]:%Y-%m-%d} to {px.index[-1]:%Y-%m-%d}, "
          f"{len(px)} closes, {n} daily returns ({'snapshot' if snapshot else 'live download'})")

    # ------------------------------------------------------------ Q1
    daily_vol = r.std()
    ann_vol = annualise_vol(daily_vol)
    print("\nQ1  Volatility, scaled from daily")
    print(f"    daily volatility        {pct(daily_vol, 3)}")
    print(f"    annualised (x sqrt 252) {pct(ann_vol)}")

    # ------------------------------------------------------------ Q2
    year_end = px.groupby(px.index.year).last()
    year_ret = year_end.pct_change().dropna()
    last_year = px.index[-1].year
    if px.index[-1] < pd.Timestamp(f"{last_year}-12-24"):
        year_ret = year_ret.drop(last_year)          # partial year
    direct = year_ret.std()
    k = len(year_ret)
    lo_ci = direct * np.sqrt((k - 1) / stats.chi2.ppf(0.975, k - 1))
    hi_ci = direct * np.sqrt((k - 1) / stats.chi2.ppf(0.025, k - 1))
    log_direct = np.log1p(year_ret).std()
    log_scaled = annualise_vol(lr.std())
    print("\nQ2  Volatility measured directly from calendar-year returns")
    print(f"    {k} full calendar years, {year_ret.index[0]}-{year_ret.index[-1]}:")
    for y, v in year_ret.items():
        print(f"      {y}  {100 * v:+7.2f}%")
    print(f"    direct annual volatility   {pct(direct)}  (95% CI {pct(lo_ci, 1)} to {pct(hi_ci, 1)})")
    print(f"    scaled from daily (Q1)     {pct(ann_vol)}")
    print(f"    ratio direct / scaled      {direct / ann_vol:.3f}")
    print(f"    same with log returns:     direct {pct(log_direct)}, scaled {pct(log_scaled)}, "
          f"ratio {log_direct / log_scaled:.3f}")

    # the square-root rule at other horizons, non-overlapping, log returns
    print("    square-root rule by horizon (log returns, non-overlapping, annualised):")
    for label, rule, per_year in [("daily", None, 252), ("weekly", "W-FRI", 52),
                                  ("monthly", "ME", 12), ("quarterly", "QE", 4)]:
        if rule is None:
            sd = lr.std()
        else:
            sd = np.log(px.resample(rule).last()).diff().dropna().std()
        print(f"      {label:<10} {pct(sd * np.sqrt(per_year))}")
    print(f"      {'annual':<10} {pct(log_direct)}")

    # link to HW3: is tomorrow independent of today?
    down = (r < 0).to_numpy()
    p_down = down.mean()
    p_dd = down[1:][down[:-1]].mean()
    ac1 = r.autocorr(1)
    ac1_sq = (r ** 2).autocorr(1)
    print(f"    HW3 link: P(down) = {p_down:.4f}, P(down | down yesterday) = {p_dd:.4f}")
    print(f"    lag-1 autocorrelation of returns {ac1:+.4f}, of squared returns {ac1_sq:+.4f}")

    # ------------------------------------------------------------ Q3
    roll = r.rolling(ROLL_WINDOW).std()
    roll_ann = annualise_vol(roll)
    hi_d, lo_d = roll.idxmax(), roll.idxmin()
    peaks = spike_episodes(roll_ann)
    print(f"\nQ3  Rolling {ROLL_WINDOW}-day volatility (dated at the window's last day)")
    print(f"    highest  {pct(roll[hi_d], 3)} daily = {pct(roll_ann[hi_d], 1)} annualised, on {hi_d:%Y-%m-%d}")
    print(f"    lowest   {pct(roll[lo_d], 3)} daily = {pct(roll_ann[lo_d], 1)} annualised, on {lo_d:%Y-%m-%d}")
    print("    largest separate spikes (peaks at least a year apart):")
    for i, (d, v) in enumerate(peaks, 1):
        print(f"      {i}. {d:%Y-%m-%d}  {pct(v, 1)}  {episode_name(d)}")
    # clustering: how persistent is the rolling level?
    high_regime = roll_ann > roll_ann.quantile(0.9)
    stay = (high_regime & high_regime.shift(ROLL_WINDOW, fill_value=False)).sum() / \
        high_regime.shift(ROLL_WINDOW, fill_value=False).sum()
    print(f"    clustering: when 60-day vol is in its top decile, it is still there "
          f"{ROLL_WINDOW} days later {pct(stay, 0)} of the time (10% if unrelated)")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(roll_ann.index, 100 * roll_ann, color=DATA_C, lw=1.2, label=f"rolling {ROLL_WINDOW}-day volatility")
    ax.axhline(100 * ann_vol, color=MUTED, ls="--", lw=1, label=f"full-sample {pct(ann_vol, 1)}")
    for i, (d, v) in enumerate(peaks[:2]):
        ax.annotate(f"{episode_name(d)}\n{d:%b %Y}, {pct(v, 0)}", xy=(d, 100 * v),
                    xytext=(25, -8), textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    for d, v in peaks[2:]:
        ax.annotate(f"{d:%b %Y}", xy=(d, 100 * v), xytext=(0, 6), textcoords="offset points",
                    fontsize=7, ha="center", color=MUTED)
    ax.set_ylabel("annualised volatility (%)")
    ax.set_title(f"SPY rolling {ROLL_WINDOW}-day volatility, {px.index[0]:%Y}-{px.index[-1]:%Y}")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "q3_rolling_vol.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------ Q4
    mu_d, sd_d = r.mean(), r.std()
    skew = stats.skew(r, bias=False)
    kurt = stats.kurtosis(r, fisher=True, bias=False)
    z = (r - mu_d) / sd_d
    tail = z.abs() > SIGMA_CUT
    n_dn, n_up = int((z < -SIGMA_CUT).sum()), int((z > SIGMA_CUT).sum())
    print("\nQ4  Shape of daily returns")
    print(f"    mean {pct(mu_d, 4)}, standard deviation {pct(sd_d, 3)}")
    print(f"    skewness          {skew:+.3f}")
    print(f"    excess kurtosis   {kurt:+.2f}")
    print(f"    days beyond {SIGMA_CUT:.0f} sd    {int(tail.sum())}  ({n_dn} down, {n_up} up;"
          f" {SIGMA_CUT:.0f} sd = {pct(SIGMA_CUT * sd_d)} a day)")
    print(f"    worst day {pct(r.min())} on {r.idxmin():%Y-%m-%d} ({z.min():.1f} sd), "
          f"best day {pct(r.max())} on {r.idxmax():%Y-%m-%d} (+{z.max():.1f} sd)")
    tail_years = pd.Series(r.index[tail].year).value_counts().sort_index()
    print("    4-sd days by year: " + ", ".join(f"{y}: {c}" for y, c in tail_years.items()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    grid = np.linspace(r.min() * 1.05, r.max() * 1.05, 600)
    bins = np.linspace(r.min(), r.max(), 160)
    for ax, logy in zip(axes, [False, True]):
        ax.hist(r, bins=bins, density=True, color=DATA_C, alpha=0.75, label="SPY daily returns")
        ax.plot(grid, stats.norm.pdf(grid, mu_d, sd_d), color=NORM_C, lw=1.6,
                label=f"normal, same mean and sd")
        for s in (-SIGMA_CUT, SIGMA_CUT):
            ax.axvline(mu_d + s * sd_d, color=MUTED, ls=":", lw=1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
        ax.set_xlabel("daily return")
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-2, None)
            ax.set_title("Log density: the tails")
        else:
            ax.set_xlim(-0.06, 0.06)
            ax.set_title("Linear density, middle of the distribution")
            ax.legend(frameon=False, loc="upper left")
    axes[1].text(0.02, 0.95, f"dotted: ±{SIGMA_CUT:.0f} sd\nskew {skew:+.2f}\nexcess kurtosis {kurt:+.1f}",
                 transform=axes[1].transAxes, va="top", fontsize=8)
    fig.suptitle(f"SPY daily returns against a fitted normal, {px.index[0]:%Y}-{px.index[-1]:%Y}")
    fig.tight_layout()
    fig.savefig(HERE / "q4_return_histogram.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------ Q5
    p_tail = 2 * stats.norm.sf(SIGMA_CUT)
    expected = n * p_tail
    actual = int(tail.sum())
    print(f"\nQ5  {SIGMA_CUT:.0f}-sd days, normal vs actual")
    print(f"    P(|z| > {SIGMA_CUT:.0f}) under a normal = {p_tail:.3e}, i.e. one day in {1 / p_tail:,.0f}"
          f" ({1 / p_tail / TRADING_DAYS:,.0f} years)")
    print(f"    expected under normality  {expected:.2f} days over {n} days")
    print(f"    actual                    {actual} days")
    print(f"    ratio actual / expected   {actual / expected:.0f}x")
    for cut in (3, 5, 6):
        e = n * 2 * stats.norm.sf(cut)
        a = int((z.abs() > cut).sum())
        print(f"    at {cut} sd: expected {e:.4g}, actual {a}, ratio {a / e:,.0f}x")

    # ------------------------------------------------------------ Q6
    print("\nQ6  Loss and the gain needed to get back to even")
    for L in LOSSES:
        print(f"    lose {100 * L:>3.0f}%  ->  need +{100 * recovery_needed(L):,.1f}%")
    peak = px.cummax()
    dd = px / peak - 1
    trough = dd.idxmin()
    top = px[:trough].idxmax()
    back = px[trough:][px[trough:] >= peak[trough]]
    back_d = back.index[0] if len(back) else None
    print(f"    SPY max drawdown (total return) {pct(dd.min(), 1)}, {top:%Y-%m-%d} to {trough:%Y-%m-%d};"
          f" needed +{100 * recovery_needed(-dd.min()):.0f}%, back to the old high on "
          f"{back_d:%Y-%m-%d}" if back_d is not None else "    (not yet recovered)")
    fig, ax = plt.subplots(figsize=(7, 4))
    curve = np.linspace(0, 0.92, 300)
    ax.plot(100 * curve, 100 * recovery_needed(curve), color=DATA_C, label="gain needed to recover")
    ax.plot(100 * curve, 100 * curve, color=MUTED, ls="--", lw=1, label="gain = loss (the naive guess)")
    for L in LOSSES:
        g = recovery_needed(L)
        ax.plot(100 * L, 100 * g, "o", color=NORM_C)
        ax.annotate(f"-{100 * L:.0f}% → +{100 * g:,.0f}%", (100 * L, 100 * g),
                    xytext=(8, -12) if L < 0.2 else (-8, 8),
                    textcoords="offset points", ha="left" if L < 0.2 else "right", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlim(0, 95)
    ax.set_ylim(5, 2000)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:,.0f}%"))
    ax.set_xlabel("loss (%)")
    ax.set_ylabel("gain needed (%), log scale")
    ax.set_title("The asymmetry of compounding: gain needed = loss / (1 - loss)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "q6_recovery.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------ Q7
    m_log, s_log = lr.mean(), lr.std()
    vol = annualise_vol(s_log)
    drift = m_log * TRADING_DAYS + 0.5 * vol ** 2      # arithmetic drift mu
    gbm = simulate_paths(START_VALUE, drift, vol, paths=N_PATHS, seed=SEED)
    print(f"\nQ7  GBM, {N_PATHS:,} one-year paths from {money(START_VALUE)} (seed {SEED})")
    print(f"    estimated from daily log returns: drift mu = {pct(drift)} a year, "
          f"volatility = {pct(vol)} a year (median growth mu - vol^2/2 = {pct(drift - vol ** 2 / 2)})")

    def summary(f):
        return {"mean": f.mean(), "median": np.median(f), "p5": np.quantile(f, 0.05),
                "p95": np.quantile(f, 0.95)}

    sg = summary(gbm)
    print(f"    mean final value     {money(sg['mean'])}   (theory {money(START_VALUE * np.exp(drift))})")
    print(f"    median final value   {money(sg['median'])}   (theory {money(START_VALUE * np.exp(drift - vol ** 2 / 2))})")
    print(f"    5th percentile       {money(sg['p5'])}")
    print(f"    95th percentile      {money(sg['p95'])}")
    print(f"    paths ending below the mean: {pct((gbm < sg['mean']).mean(), 1)}; "
          f"below the start: {pct((gbm < START_VALUE).mean(), 1)}")

    # ------------------------------------------------------------ Q8
    def risk(f):
        ret = f / START_VALUE - 1.0
        v, cv = var(ret), cvar(ret)
        return {"var": -v, "cvar": -cv, "var$": -v * START_VALUE, "cvar$": -cv * START_VALUE}

    rg = risk(gbm)
    print(f"\nQ8  One-year risk at {CONFIDENCE:.0%}, GBM (loss measured from the {money(START_VALUE)} start)")
    print(f"    VaR   {money(rg['var$'])}  = {pct(rg['var'])} of the portfolio")
    print(f"    CVaR  {money(rg['cvar$'])}  = {pct(rg['cvar'])} of the portfolio")

    # ------------------------------------------------------------ Q9
    boot = bootstrap_paths(START_VALUE, r, paths=N_PATHS, seed=SEED)
    block = bootstrap_paths(START_VALUE, r, paths=N_PATHS, seed=SEED, block=BLOCK)
    hist_1y = (px.shift(-TRADING_DAYS) / px - 1).dropna()     # every overlapping 252-day window
    sb, sk = summary(boot), summary(block)
    rb, rk = risk(boot), risk(block)
    rh = {"var": -var(hist_1y), "cvar": -cvar(hist_1y)}
    rh["var$"], rh["cvar$"] = rh["var"] * START_VALUE, rh["cvar"] * START_VALUE
    print(f"\nQ9  Same thing, resampling actual daily returns ({N_PATHS:,} paths, seed {SEED})")
    head = f"    {'':<22}{'GBM (normal)':>16}{'bootstrap':>16}{f'block boot ({BLOCK}d)':>18}{'history':>14}"
    print(head)
    print(f"    {'':<22}{'':>16}{'(iid days)':>16}{'(extra)':>18}{'(252d windows)':>14}")
    rows = [("mean final", "mean"), ("median final", "median"), ("5th percentile", "p5"),
            ("95th percentile", "p95")]
    hs = {"mean": START_VALUE * (1 + hist_1y.mean()), "median": START_VALUE * (1 + hist_1y.median()),
          "p5": START_VALUE * (1 + hist_1y.quantile(0.05)), "p95": START_VALUE * (1 + hist_1y.quantile(0.95))}
    for label, key in rows:
        print(f"    {label:<22}{money(sg[key]):>16}{money(sb[key]):>16}{money(sk[key]):>18}{money(hs[key]):>14}")
    for label, key, kp in [("VaR 95%", "var$", "var"), ("CVaR 95%", "cvar$", "cvar")]:
        print(f"    {label + ' ($)':<22}{money(rg[key]):>16}{money(rb[key]):>16}{money(rk[key]):>18}{money(rh[key]):>14}")
        print(f"    {label + ' (% of start)':<22}{pct(rg[kp]):>16}{pct(rb[kp]):>16}{pct(rk[kp]):>18}{pct(rh[kp]):>14}")
    for name, f in [("GBM", gbm), ("bootstrap", boot), ("block bootstrap", block)]:
        lr_1y = np.log(f / START_VALUE)
        print(f"    {name:<16} one-year log return: sd {pct(lr_1y.std())}, "
              f"skew {stats.skew(lr_1y):+.2f}, excess kurtosis {stats.kurtosis(lr_1y):+.2f}, "
              f"worst path {pct(f.min() / START_VALUE - 1)}")
    print(f"    history: {len(hist_1y)} overlapping 252-day windows "
          f"(about {len(hist_1y) / TRADING_DAYS:.0f} independent years); worst {pct(hist_1y.min())} "
          f"from {hist_1y.idxmin():%Y-%m-%d}")
    # Monte Carlo noise in the tail estimates
    reps = [risk(simulate_paths(START_VALUE, drift, vol, paths=N_PATHS, seed=s)) for s in range(20)]
    repb = [risk(bootstrap_paths(START_VALUE, r, paths=N_PATHS, seed=s)) for s in range(20)]
    print("    seed-to-seed spread over 20 seeds (sd of the % figure):")
    print(f"      GBM       VaR {pct(np.std([x['var'] for x in reps]))}, CVaR {pct(np.std([x['cvar'] for x in reps]))}")
    print(f"      bootstrap VaR {pct(np.std([x['var'] for x in repb]))}, CVaR {pct(np.std([x['cvar'] for x in repb]))}")
    print(f"      mean over seeds: GBM VaR {pct(np.mean([x['var'] for x in reps]))} / CVaR "
          f"{pct(np.mean([x['cvar'] for x in reps]))}; bootstrap VaR {pct(np.mean([x['var'] for x in repb]))}"
          f" / CVaR {pct(np.mean([x['cvar'] for x in repb]))}")
    for key in ("var", "cvar"):
        d = np.array([b[key] - g[key] for b, g in zip(repb, reps)])
        print(f"      bootstrap minus GBM, {key.upper():<4}: {100 * d.mean():+.2f} points "
              f"(standard error {100 * d.std(ddof=1) / np.sqrt(len(d)):.2f})")
    print(f"    daily excess kurtosis / 252 = {kurt / TRADING_DAYS:+.3f}: what iid summing leaves of the fat tails")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(min(gbm.min(), boot.min(), block.min()),
                       np.quantile(np.r_[gbm, boot, block], 0.999), 120)
    ax = axes[0]
    ax.hist(gbm, bins=bins, color=NORM_C, alpha=0.55, label="GBM (normal)")
    ax.hist(boot, bins=bins, color=BOOT_C, alpha=0.55, label="bootstrap (iid days)")
    ax.axvline(sg["median"], color="k", lw=1, label=f"GBM median {money(sg['median'])}")
    ax.axvline(sg["mean"], color="k", ls="--", lw=1, label=f"GBM mean {money(sg['mean'])}")
    ax.axvline(START_VALUE, color=MUTED, ls=":", lw=1, label="start $100,000")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x / 1000:.0f}k"))
    ax.set_title("Final value after one year, 10,000 paths")
    ax.set_xlabel("final portfolio value")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    table = []
    # each method as one-year returns; history is every overlapping 252-day window
    for ret, c, lab, rr, ls in [(gbm / START_VALUE - 1, NORM_C, "GBM", rg, "-"),
                                (boot / START_VALUE - 1, BOOT_C, "bootstrap", rb, "-"),
                                (block / START_VALUE - 1, DATA_C, f"block bootstrap ({BLOCK}d)", rk, "-"),
                                (hist_1y.to_numpy(), "k", "history, 252-day windows", rh, "--")]:
        q = np.sort(ret)[: int(0.10 * len(ret))]
        ax.plot(np.arange(1, len(q) + 1) / len(ret) * 100, 100 * q, color=c, ls=ls,
                lw=1.4 if ls == "--" else 2, label=lab)
        table.append(f"{lab.split(' (')[0].split(',')[0]:<16}{pct(rr['var'], 1):>7}{pct(rr['cvar'], 1):>8}")
    ax.axvline(5, color=MUTED, ls=":", lw=1)
    ax.text(0.97, 0.40, f"{'95%':<16}{'VaR':>7}{'CVaR':>8}\n" + "\n".join(table),
            transform=ax.transAxes, ha="right", va="top", fontsize=7, family="monospace")
    ax.set_xlabel("worst x% of outcomes (dotted line: the 5% VaR cut)")
    ax.set_ylabel("one-year return (%)")
    ax.set_title("Left 10% of one-year outcomes")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "q7_q9_monte_carlo.png", dpi=160)
    plt.close(fig)

    print("\nChecks: daily vol in 0.7%-1.3%:", 0.007 <= daily_vol <= 0.013,
          "| annual in 12%-20%:", 0.12 <= ann_vol <= 0.20,
          "| recovery(50%) = +100%:", recovery_needed(0.5) == 1.0,
          "| CVaR worse than VaR (all methods):",
          all(x["cvar"] >= x["var"] for x in (rg, rb, rk, rh)))


if __name__ == "__main__":
    main()
