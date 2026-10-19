"""Homework 4 self-check.   python check_hw04.py

The important checks here are the self-consistency ones. A backtester with an
off-by-one error still produces a plausible-looking number, and the only way to
catch it is to run a strategy whose answer you already know.

Every series below is built here, so these run with no network and give the same
answer every time.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _check import Checker
import numpy as np

c = Checker("Homework 4 self-check", "hw04_starter.py")
m = c.load(__file__)

LOOKBACK = 60


def _no_lookahead(m):
    """Detect a backtester that hands the strategy the day it is predicting.

    Feed it a strictly increasing price series, so a price doubles as its own
    index. A probe strategy records the newest price it is ever shown. A correct
    backtester never shows the strategy the final price, because that price only
    ever exists as a "tomorrow". A lookahead backtester shows it.
    """
    ladder = [100.0 + i for i in range(400)]
    highest_seen = [float("-inf")]

    def probe(window):
        highest_seen[0] = max(highest_seen[0], float(max(window)))
        return True

    m.backtest(ladder, probe)
    if highest_seen[0] == float("-inf"):
        return False                      # the strategy was never called at all
    return highest_seen[0] < ladder[-1]


rng = np.random.default_rng(11)
prices = list(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 1200))))
# buy and hold over exactly the days the backtest trades: the first decision is
# made on the 60th close, index 59, and the last return runs to the final close
index_total = prices[-1] / prices[LOOKBACK - 1] - 1
_one_day_late = prices[-1] / prices[LOOKBACK] - 1

c.exists("backtest is defined", "backtest")
c.exists("a strategy function is defined", "always_hold")

c.check("always-hold matches buy-and-hold over the days traded",
        lambda: m.backtest(prices, m.always_hold)["total_return"], index_total, tol=1e-6,
        note=f"expected prices[-1]/prices[59] - 1. If you got {_one_day_late:.6g} instead, your window "
             f"ends yesterday rather than today: you are withholding today's close, which is known when "
             f"the decision is made. That is a start-index error, not lookahead, and the lookahead probe "
             f"below is the check that tests for lookahead.")
c.check("always-cash returns exactly zero",
        lambda: m.backtest(prices, m.always_cash)["total_return"], 0.0, tol=1e-12,
        note="holding cash every day cannot make or lose anything")

c.assert_true("the strategy never sees the day it is predicting",
              lambda: _no_lookahead(m),
              note="a probe strategy was shown the windows your backtester hands out, and one of them "
                   "contained the price it was supposed to be predicting.")

# ---------------------------------------------------------------- annualising
# 20 basis points a day, every day, so the annualised figure is known in closed
# form and no simulation is involved.
_G = 0.002
_steady = [100.0 * (1 + _G) ** i for i in range(400)]

c.check("annualised return compounds at 252 trading days a year",
        lambda: m.backtest(_steady, m.always_hold)["annualised_return"],
        (1 + _G) ** 252 - 1, tol=1e-6,
        note="value ** (252 / days) - 1, where days is the number of returns applied. Using 365, or "
             "multiplying the daily mean by 252 instead of compounding, both land somewhere else.")
c.check("annualised return of always-cash is exactly zero",
        lambda: m.backtest(_steady, m.always_cash)["annualised_return"], 0.0, tol=1e-12,
        note="cash compounds at nothing, however many years you do it for")

# ---------------------------------------------------------------- drawdown, exposure
# up 20%, then down 30% from that peak, then flat: the worst fall from a peak is
# 30% whatever happens afterwards.
_hump = ([100.0 * (1.2 ** (i / 100)) for i in range(101)]
         + [120.0 * (0.7 ** (i / 100)) for i in range(1, 101)]
         + [84.0] * 100)

c.check("max drawdown is the worst fall from a peak",
        lambda: m.backtest(_hump, m.always_hold)["max_drawdown"], -0.30, tol=0.005,
        note="the series rises 20% then falls 30% off that peak. Measure the fall from the running "
             "maximum of PORTFOLIO VALUE, not from the first value and not from the final one.")
c.assert_true("max drawdown is negative or zero, never positive",
              lambda: m.backtest(prices, m.always_hold)["max_drawdown"] <= 0.0,
              note="a drawdown is a fall from a peak, so it cannot be a positive number")

c.check("fraction invested is 1.0 when always holding",
        lambda: m.backtest(prices, m.always_hold)["fraction_invested"], 1.0, tol=1e-12,
        note="every day was invested, so the fraction is 1, not the count of days")
c.check("fraction invested is 0.0 when always in cash",
        lambda: m.backtest(prices, m.always_cash)["fraction_invested"], 0.0, tol=1e-12,
        note="no day was invested")

# ---------------------------------------------------------------- the class rule
# windows of the right length whose last five days are unambiguous. The rule holds
# UNLESS the 5-day return to today is positive, so a flat five days means hold.
def _window(tail):
    return [100.0] * (LOOKBACK - len(tail)) + list(tail)

_rose = _window([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
_fell = _window([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
_flat = _window([100.0] * 6)
# down over the five days but up on the last one: the rule reads the five-day
# move, not yesterday-to-today
_zigzag = _window([100.0, 96.0, 95.0, 94.0, 93.0, 97.0])

c.assert_true("five_day_reversal goes to cash after five up days",
              lambda: m.five_day_reversal(_rose) is False,
              note="the 5-day return to today is positive, so the rule says cash. Returning True here "
                   "means the condition is inverted.")
c.assert_true("five_day_reversal holds after five down days",
              lambda: m.five_day_reversal(_fell) is True,
              note="the 5-day return to today is negative, so the rule holds")
c.assert_true("five_day_reversal holds when the five days are flat",
              lambda: m.five_day_reversal(_flat) is True,
              note="the rule is 'unless positive', and zero is not positive")
c.assert_true("five_day_reversal reads five days, not one",
              lambda: m.five_day_reversal(_zigzag) is True,
              note="this window fell over five days but rose on the last one. Comparing window[-1] to "
                   "window[-2] gives cash here; comparing it to window[-6] gives hold.")

# ---------------------------------------------------------------- the random control
c.assert_true("a random 70% strategy lands between cash and the index",
              lambda: 0.0 <= m.backtest(prices, m.make_random_hold(0.7, seed=0))["total_return"]
                      <= index_total * 1.4,
              note="being invested most of the time should land most of the way to the index")
c.assert_true("two random strategies built with the same seed agree exactly",
              lambda: (m.backtest(prices, m.make_random_hold(0.7, seed=4))["total_return"]
                       == m.backtest(prices, m.make_random_hold(0.7, seed=4))["total_return"]),
              note="each call to make_random_hold must build its own generator. Sharing one generator "
                   "across strategies, which is what a default argument does, makes the same backtest "
                   "return a different number every time you run it.")
c.assert_true("a 70% strategy is invested roughly 70% of the time",
              lambda: 0.63 <= m.backtest(prices, m.make_random_hold(0.7, seed=1))["fraction_invested"] <= 0.77,
              note="over 1,140 days the realised fraction should sit close to 0.7")

# ---------------------------------------------------------------- the split
# a business-day series with a known split date, so the overlap can be counted
import pandas as pd
_idx = pd.bdate_range("2011-01-03", periods=400)
_ser = pd.Series(np.linspace(100.0, 200.0, 400), index=_idx)
_SPLIT = "2011-09-09"


def _split(**kw):
    return m.load_and_split(split_date=_SPLIT, prices=_ser, **kw)


c.assert_true("load_and_split can split a series it was handed",
              lambda: len(_split()[0]) > 0 and len(_split()[1]) > 0,
              note="take an optional prices argument so this can be checked without a download")
c.assert_true("training ends on the split date",
              lambda: str(_split()[0].index[-1].date()) == _SPLIT,
              note=f"the last training close must be {_SPLIT}, which is a Friday and a trading day")
c.assert_true("the test set carries 60 days of window history",
              lambda: len(_split()[1]) - len(_ser) + len(_split()[0]) == LOOKBACK,
              note="test must start 60 rows before the first out-of-sample day, so the 60-day window is "
                   "full on that day. Without the overlap your out-of-sample period starts three months "
                   "late and you will not be told.")
c.assert_true("the test set runs to the end of the data",
              lambda: _split()[1].index[-1] == _ser.index[-1],
              note="nothing after the split date may be dropped")
c.assert_true("the first out-of-sample return is earned after the split date",
              lambda: _split()[1].index[LOOKBACK - 1] == _split()[0].index[-1],
              note="the last day of the test set's warm-up is the split close itself, so the first "
                   "return the test-set backtest earns runs from the split date to the next session")

sys.exit(c.report())
