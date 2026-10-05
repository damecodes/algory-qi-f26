"""Homework 3 starter — Algory QI Education, Fall 2026

Fill in every function marked TODO. Keep the names: the checker looks for them.
Run `python check_hw03.py` as you go.
"""
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- Q1
def simulate_dice(trials, seed=0):
    """Pick a 4-sided or 6-sided die at random, roll it, repeat.

    Return the estimated P(picked the 4-sided die | rolled a 1).
    The exact answer is 0.6. Take a seed so your result is reproducible.
    """
    rng = np.random.default_rng(seed)
    sides = rng.choice([4, 6], size=trials)   # which die was picked
    rolls = rng.integers(1, sides + 1)        # roll it: uniform on 1..sides
    ones = rolls == 1
    return np.mean(sides[ones] == 4)


# ---------------------------------------------------------------- Q2
def simulate_coins(trials, seed=0):
    """Flip three fair coins, get paid (heads x tails). Return the mean payout.

    The exact answer is 1.5.
    """
    rng = np.random.default_rng(seed)
    heads = rng.integers(0, 2, size=(trials, 3)).sum(axis=1)
    tails = 3 - heads
    return np.mean(heads * tails)


# ---------------------------------------------------------------- Q5/Q6
def p_down(returns):
    """Fraction of days with a negative return. Count them yourself."""
    down, total = count_down(returns)
    return down / total if total else float("nan")


def count_down(returns):
    """Return (days with a negative return, total days), counted by hand."""
    down = 0
    total = 0
    for r in returns:
        total += 1
        if r < 0:
            down += 1
    return down, total


def p_down_given_down(returns):
    """P(tomorrow is down | today was down), counted directly from the series."""
    down_next, down_today = count_down_after_down(returns)
    return down_next / down_today if down_today else float("nan")


def count_down_after_down(returns):
    """Return (down days followed by a down day, down days that have a next day).

    The final day is skipped as "today" because it has no tomorrow.
    """
    down_next = 0
    down_today = 0
    for i in range(len(returns) - 1):
        if returns[i] < 0:
            down_today += 1
            if returns[i + 1] < 0:
                down_next += 1
    return down_next, down_today


def p_down_given_big_drop(returns, threshold=-0.02):
    """P(tomorrow is down | today fell more than the threshold).

    Also report how many days the estimate uses. A handful of days is a much
    weaker claim than thousands, and the count is what tells a reader which of
    the two this is.
    """
    down_next, big_drops = count_down_after_big_drop(returns, threshold)
    return down_next / big_drops if big_drops else float("nan")


def count_down_after_big_drop(returns, threshold=-0.02):
    """Return (big-drop days followed by a down day, big-drop days with a next day).

    A big drop is a return below the threshold. The final day is skipped as
    "today" because it has no tomorrow.
    """
    down_next = 0
    big_drops = 0
    for i in range(len(returns) - 1):
        if returns[i] < threshold:
            big_drops += 1
            if returns[i + 1] < 0:
                down_next += 1
    return down_next, big_drops


# ---------------------------------------------------------------- Q7
def expected_present_value(cash_flows, rate, survival_prob):
    """Present value where the company survives EACH year with survival_prob.

    Year 1 is certain. Year 2 arrives with probability survival_prob, year 3
    with survival_prob squared, and so on. This is a yearly hazard rate.

    Note this is deliberately more general than the Session 3 slide, which had a
    single shutdown event after year 1 and came to 16.98. A yearly 50% survival
    is a harsher assumption and gives 15.10. Getting 16.98 here means you applied
    the probability once instead of compounding it.
    """
    discount_factors = [(1 + rate) ** i for i in range(1, len(cash_flows) + 1)]
    arrival_probs = [survival_prob ** i for i in range(len(cash_flows))]
    epv = sum(cf * p / df for cf, p, df in zip(cash_flows, arrival_probs, discount_factors))
    return round(epv, 2)


def main():
    sim = simulate_dice(100_000, seed=42)
    print(f"Q1  simulated P(4-sided | rolled a 1) = {sim:.4f}  (seed=42)")
    print(f"    exact = 0.6000, difference = {sim - 0.6:+.4f}")
    payout = simulate_coins(100_000, seed=42)
    print(f"Q2  simulated expected payout = {payout:.4f}  (seed=42)")
    print(f"    exact = 1.5000, difference = {payout - 1.5:+.4f}")
    counts = [100, 1_000, 10_000, 100_000]
    estimates = [simulate_coins(n, seed=42) for n in counts]
    for n, est in zip(counts, estimates):
        print(f"Q3  {n:>7,} trials: estimate = {est:.4f}")
    plt.figure(figsize=(7, 4))
    plt.plot(counts, estimates, "o-", label="simulated estimate")
    plt.axhline(1.5, color="red", linestyle="--", label="exact = 1.5")
    plt.xscale("log")
    plt.xlabel("number of trials")
    plt.ylabel("estimated expected payout")
    plt.title("Three-coin payout: estimate vs. trials (seed=42)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("q3_coin_convergence.png", dpi=150)
    close = yf.download("SPY", period="10y", auto_adjust=True, progress=False)["Close"]
    close = close.squeeze().dropna()
    returns = close.pct_change().dropna().to_numpy()
    print(f"Q4  {len(close)} trading days of prices, {len(returns)} daily returns, "
          f"mean daily return = {returns.mean():.6f}")

    down, total = count_down(returns)
    print(f"Q5  P(down) = {p_down(returns):.4f}  ({down} down days out of {total})")
    down_next, down_today = count_down_after_down(returns)
    print(f"    P(down tomorrow | down today) = {p_down_given_down(returns):.4f}  "
          f"({down_next} of {down_today} down days were followed by a down day)")

    down_next, big_drops = count_down_after_big_drop(returns)
    print(f"Q6  P(down tomorrow | down more than 2% today) = "
          f"{p_down_given_big_drop(returns):.4f}  "
          f"({down_next} of {big_drops} big-drop days were followed by a down day)")
    print("Q7  EPV =", expected_present_value([10, 10, 10], 0.10, 0.5))


if __name__ == "__main__":
    main()
