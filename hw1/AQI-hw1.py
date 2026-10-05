import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Step 6
print("Ticker, Trading Days, First Date, Last Date: \n")
tickers = ["IAG", "SPY"]

# auto_adjust=True, includes dividends
data = yf.download(tickers, period="1y", auto_adjust=True, progress=False)["Close"]
data = data.dropna()

print("Tickers:", list(data.columns))
print("Trading days:", len(data))
print("First date:", data.index[0].date())
print("Last date:", data.index[-1].date())

# Step 7
print("Annualized Returns and Volatility: \n")

# Calculate daily returns
returns = data.pct_change().dropna()

for ticker in data.columns:
    last_close = data[ticker].iloc[-1]
    total_return = data[ticker].iloc[-1] / data[ticker].iloc[0] - 1
    ann_vol = returns[ticker].std() * np.sqrt(252)

    print(f"{ticker}")
    print(f"  Last close:        ${last_close:,.2f}")
    print(f"  Return over year:  {total_return:.2%}")
    print(f"  Annualised vol:    {ann_vol:.2%}")

# Step 8
print("\nPlotting rebased prices: \n")

# Plot rebased prices 
rebased = data / data.iloc[0] * 100

fig, ax = plt.subplots(figsize=(10, 6))

for ticker in rebased.columns:
    ax.plot(rebased.index, rebased[ticker], label=ticker, linewidth=1.5)

ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)

ax.set_title("IAG vs SPY, rebased to 100")
ax.set_xlabel("Date")
ax.set_ylabel("Value of $100 invested")
ax.legend()
ax.grid(alpha=0.3)

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("hw01_rebased.png", dpi=150)
plt.show()

# Step 9
print("Biggest one day move:")

iag = returns["IAG"]

biggest_date = iag.abs().idxmax()
biggest_move = iag.loc[biggest_date]

print(f"Biggest single-day move: {biggest_date.date()}, {biggest_move:+.2%}")
print(f"SPY that day:            {returns['SPY'].loc[biggest_date]:+.2%}")