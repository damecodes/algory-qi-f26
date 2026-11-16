"""Homework 8 self-check.   python check_hw08.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _check import Checker
import numpy as np

c = Checker("Homework 8 self-check", "hw08_starter.py")
m = c.load(__file__)


rng = np.random.default_rng(8)
PRICES = list(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, 3000))))

c.exists("n_day_returns is defined", "n_day_returns")
c.exists("price_from_history is defined", "price_from_history")

c.assert_true("overlapping 21-day windows give about len(prices) - 21 of them",
              lambda: abs(len(m.n_day_returns(PRICES, 21)) - (len(PRICES) - 21)) <= 2,
              note="one window per starting day that has 21 days after it")
c.assert_true("a 1-day window reproduces the daily returns",
              lambda: abs(np.std(m.n_day_returns(PRICES, 1))
                          - np.std(np.diff(PRICES) / np.array(PRICES[:-1]))) < 1e-6,
              note="an N of 1 is just the daily return series")
c.assert_true("longer windows have wider spread than shorter ones",
              lambda: np.std(m.n_day_returns(PRICES, 63)) > np.std(m.n_day_returns(PRICES, 5)),
              note="more days is more accumulated movement")

c.assert_true("a deep in-the-money call prices near its intrinsic value",
              lambda: abs(m.price_from_history(PRICES, 100.0, 50.0, 21, "call") - 50.0) < 6.0,
              note="with the strike far below spot the option is essentially the stock")
c.assert_true("a far out-of-the-money call is nearly worthless",
              lambda: m.price_from_history(PRICES, 100.0, 300.0, 21, "call") < 1.0,
              note="no 21-day window in the sample gets anywhere near tripling")
c.assert_true("option prices are never negative",
              lambda: all(m.price_from_history(PRICES, 100.0, k, 21, kind) >= -1e-9
                          for k in (60, 90, 100, 110, 160) for kind in ("call", "put")),
              note="an option confers a right, never an obligation, so it cannot be worth less than zero")
c.assert_true("a call is worth more than a put at a strike below spot",
              lambda: (m.price_from_history(PRICES, 100.0, 85.0, 21, "call")
                       > m.price_from_history(PRICES, 100.0, 85.0, 21, "put")),
              note="the call is already deep in the money and the put is not")
c.assert_true("a longer-dated contract is worth at least as much",
              lambda: (m.price_from_history(PRICES, 100.0, 100.0, 63, "call")
                       >= m.price_from_history(PRICES, 100.0, 100.0, 21, "call") - 1e-9),
              note="more time cannot make an option less valuable")

sys.exit(c.report())
