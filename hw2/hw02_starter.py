"""Homework 2 starter — Algory QI Education, Fall 2026

Fill in every function marked TODO. Do not rename them: the checker looks for
these exact names. Run `python check_hw02.py` before you submit.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- Q1
def present_value(cash_flows, rate):
    """Present value of a list of cash flows, the first arriving in one year.

    present_value([10, 15, 20], 0.10) -> 36.51
    """
    discount_factors = [(1 + rate) ** i for i in range(1, len(cash_flows) + 1)]
    present_value = sum(cf / df for cf, df in zip(cash_flows, discount_factors))
    return round(present_value, 2)


# ---------------------------------------------------------------- Q2
def bond_price(face, coupon_rate, years, market_rate):
    """Price of a bond paying an annual coupon and repaying face at maturity.

    The final year pays the coupon AND the face value. That is the usual bug.
    bond_price(1000, 0.04, 10, 0.04) -> exactly 1000.0
    """
    cash_flows = [face * coupon_rate] * years
    cash_flows[-1] += face
    return present_value(cash_flows, market_rate)


# ---------------------------------------------------------------- Q4/Q5
# Three sectors: IAG = gold mining (materials), JPM = banking (financials),
# JNJ = pharma (health care). SPY is the market benchmark.
TICKERS = ["IAG", "JPM", "JNJ"]
BENCHMARK = "SPY"
 
 
def download_closes(tickers, years=5):
    """Daily closing prices for each ticker over the last `years` years.
 
    auto_adjust=True gives closes adjusted for splits and dividends, which is
    what Q5's returns should be computed from. Returns one column per ticker.
    """
    data = yf.download(tickers, period=f"{years}y", auto_adjust=True,
                       progress=False)["Close"]
    missing = [t for t in tickers if t not in data or data[t].dropna().empty]
    if missing:
        raise RuntimeError(f"No data downloaded for {missing}; check the "
                           "ticker symbols and your internet connection.")
    return data[tickers]
 
 
TRADING_DAYS = 252
 
 
def daily_returns(prices):
    """Simple daily returns: P_t / P_{t-1} - 1. Missing prices are dropped first."""
    return prices.dropna().pct_change().dropna()
 
 
def annualised_return(prices):
    """Annualised return from a price series, using 252 trading days."""
    # Compound (geometric) rate: total growth over N daily returns, scaled to
    # 252 days a year. This is the rate that actually turns P_start into P_end.
    p = prices.dropna()
    n_returns = len(p) - 1
    total_growth = p.iloc[-1] / p.iloc[0]
    return total_growth ** (TRADING_DAYS / n_returns) - 1
 
 
def annualised_volatility(prices):
    """Annualised standard deviation of daily returns."""
    # Daily std (sample, ddof=1) scaled by sqrt(252): variance grows with time.
    return daily_returns(prices).std() * np.sqrt(TRADING_DAYS)
 
 
def beta(stock_prices, market_prices):
    """Beta of a stock against the market.
 
    Covariance of the two RETURN series divided by the variance of the market's.
    Computing this on prices instead of returns is a common and silent error.
    beta(spy, spy) -> 1.0
    """
    # Line the two price series up on the dates both traded, THEN take returns,
    # so each stock return is paired with the market return for the same day.
    both = pd.concat([stock_prices, market_prices], axis=1, join="inner").dropna()
    rets = both.pct_change().dropna()
    stock_r, market_r = rets.iloc[:, 0], rets.iloc[:, 1]
    # Same ddof (1) in numerator and denominator, so it cancels out.
    return stock_r.cov(market_r) / market_r.var()
 
 
# ---------------------------------------------------------------- your answers
def main():
    """Everything the assignment asks you to print goes here."""
    print("Q1  present_value([10, 15, 20], 0.10) =", present_value([10, 15, 20], 0.10))
    print("Q2  bond_price(1000, 0.04, 10, 0.04) =", bond_price(1000, 0.04, 10, 0.04))
 
    # Q3: 10-year, 4% annual coupon bond at three market rates
    print("Q3  10-year 4% coupon bond, face 1000")
    for label, r in [("2%", 0.02), ("4%", 0.04), ("4.96% (10Y Treasury)", 0.0496)]:
        print(f"    market rate {label:>22}: {bond_price(1000, 0.04, 10, r):8.2f}")
 
    rates = np.linspace(0, 0.10, 201)
    prices = [bond_price(1000, 0.04, 10, r) for r in rates]
    marked = [0.02, 0.04, 0.0496]
 
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rates * 100, prices, color="#2a6fdb", linewidth=2)
    ax.scatter([r * 100 for r in marked],
               [bond_price(1000, 0.04, 10, r) for r in marked],
               s=40, color="#2a6fdb", zorder=3)
    for r in marked:
        p = bond_price(1000, 0.04, 10, r)
        ax.annotate(f"{r * 100:g}%: {p:,.2f}", (r * 100, p),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.axhline(1000, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Market rate (%)")
    ax.set_ylabel("Bond price ($)")
    ax.set_title("10-year 4% coupon bond: price vs market rate")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig("q3_bond_price_vs_rate.png", dpi=150)
    plt.close(fig)
    print("    plot saved to q3_bond_price_vs_rate.png")
 
    # Convexity check: the gain from a 2-point fall vs the loss from a 2-point rise
    up, down = bond_price(1000, 0.04, 10, 0.06), bond_price(1000, 0.04, 10, 0.02)
    print(f"    rate 4% -> 2%: +{down - 1000:.2f}   rate 4% -> 6%: {up - 1000:.2f}")
    print("    The curve slopes down and bows toward the origin (convex): price")
    print("    rises faster as rates fall than it drops as rates rise by the same amount.")
    # Q4: five years of daily closes for the three tickers and SPY
    closes = download_closes(TICKERS + [BENCHMARK], years=5)
    print(f"Q4  daily closes {closes.index[0]:%Y-%m-%d} to {closes.index[-1]:%Y-%m-%d}")
    for t in closes.columns:
        print(f"    {t:<4} {closes[t].dropna().shape[0]:>5} trading days")
    # Q5: return, volatility and beta for each ticker (SPY last as a check:
    # its beta against itself must be 1.00)
    spy = closes[BENCHMARK]
    stats = pd.DataFrame(
        {
            "Return": [annualised_return(closes[t]) for t in closes.columns],
            "Volatility": [annualised_volatility(closes[t]) for t in closes.columns],
            "Beta": [beta(closes[t], spy) for t in closes.columns],
        },
        index=closes.columns,
    )
    print("Q5  annualised return and volatility (252 days), beta vs SPY")
    print(stats.to_string(formatters={"Return": "{:8.2%}".format,
                                      "Volatility": "{:8.2%}".format,
                                      "Beta": "{:6.2f}".format}))
    # Q6: rank the three tickers (not SPY) by volatility and by beta, highest first
    ranked = stats.loc[TICKERS]
    by_vol = ranked["Volatility"].sort_values(ascending=False)
    by_beta = ranked["Beta"].sort_values(ascending=False)
    print("Q6  ranking by volatility:", " > ".join(by_vol.index))
    print("    ranking by beta:      ", " > ".join(by_beta.index))
    print("    rankings agree" if list(by_vol.index) == list(by_beta.index)
          else "    rankings DISAGREE")
 
    # Where the risk comes from: split each stock's variance into a market part
    # (beta * SPY vol) and a stock-specific part (everything left over).
    # R^2 = share of the stock's daily variance explained by SPY.
    spy_vol = stats.loc[BENCHMARK, "Volatility"]
    market_vol = ranked["Beta"].abs() * spy_vol
    specific_vol = np.sqrt(np.maximum(ranked["Volatility"] ** 2 - market_vol ** 2, 0))
    risk = pd.DataFrame({
        "Total vol": ranked["Volatility"],
        "Market vol": market_vol,
        "Specific vol": specific_vol,
        "R^2 vs SPY": market_vol ** 2 / ranked["Volatility"] ** 2,
    })
    print("    risk split (market part = beta x SPY vol)")
    print(risk.to_string(float_format="{:8.2%}".format))
 
 
if __name__ == "__main__":
    main()
 
