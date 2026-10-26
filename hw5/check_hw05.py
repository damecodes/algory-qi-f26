"""Homework 5 self-check.   python check_hw05.py

Every chain below is built here, so these run with no network and give the same
answer every time. The Q4 checks use a toy chain whose correct split you can
work out by hand.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _check import Checker
import numpy as np
import pandas as pd

c = Checker("Homework 5 self-check", "hw05_starter.py")
m = c.load(__file__)

def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


# ---------------------------------------------------------------- Q1
c.exists("payoff exists", "payoff")
c.check("long call, K=100, prem 6, spot 80", lambda: m.payoff("long_call", 100, 6, 80), -6.0)
c.check("short call, K=100, prem 6, spot 120", lambda: m.payoff("short_call", 100, 6, 120), -14.0)
c.check("long call, spot 120", lambda: m.payoff("long_call", 100, 6, 120), 14.0)
c.check("long put, spot 80", lambda: m.payoff("long_put", 100, 6, 80), 14.0)
c.check("long put, spot 120", lambda: m.payoff("long_put", 100, 6, 120), -6.0)
c.check("short put, spot 80", lambda: m.payoff("short_put", 100, 6, 80), -14.0)
c.check("short put, spot 120", lambda: m.payoff("short_put", 100, 6, 120), 6.0)
c.check("long call at the strike loses the premium",
        lambda: m.payoff("long_call", 100, 6, 100), -6.0)

spots = np.linspace(50, 150, 101)
c.assert_true("short is the exact negative of long, everywhere",
              lambda: all(np.allclose(m.payoff("short_" + k, 100, 6, spots),
                                      -m.payoff("long_" + k, 100, 6, spots))
                          for k in ("call", "put")))
c.assert_true("payoff accepts arrays and returns the same shape",
              lambda: m.payoff("long_put", 100, 6, spots).shape == spots.shape)
c.assert_true("long call breaks even at strike + premium",
              lambda: abs(m.payoff("long_call", 100, 6, 106)) < 1e-12)
c.assert_true("an unknown kind raises ValueError",
              lambda: _raises(lambda: m.payoff("long_straddle", 100, 6, 100), ValueError))



# ---------------------------------------------------------------- Q3
expiries = ["2026-10-30", "2026-11-20", "2026-11-27", "2026-12-18",
            "2027-03-19", "2027-06-17", "2027-12-17"]
c.assert_true("one-month pick is the expiry closest to 30 days",
              lambda: m.pick_expiry(expiries, "2026-10-25", target_days=30) == "2026-11-27")
c.assert_true("long pick is the first expiry at least 182 days out",
              lambda: m.pick_expiry(expiries, "2026-10-25", min_days=182) == "2027-06-17")

# ---------------------------------------------------------------- Q4
SPOT = 100.0
toy = pd.DataFrame({
    "strike": [50, 90, 100, 110, 150, 200],
    "bid":    [49.9, 11.0, 4.0, 1.0, 0.01, 0.0],
    "ask":    [50.1, 12.0, 4.4, 1.2, 0.03, 0.05],
})
vals = m.split_call_values(toy, SPOT)
row = lambda k: vals.set_index("strike").loc[k]

c.assert_true("zero-bid rows are dropped", lambda: 200 not in set(vals["strike"]))
c.check("mid is the average of bid and ask", lambda: row(90)["mid"], 11.5)
c.check("intrinsic of an in-the-money call is spot - strike", lambda: row(90)["intrinsic"], 10.0)
c.check("intrinsic of an out-of-the-money call is zero", lambda: row(110)["intrinsic"], 0.0)
c.check("extrinsic is mid - intrinsic", lambda: row(90)["extrinsic"], 1.5)
c.assert_true("intrinsic + extrinsic equals mid on every row",
              lambda: np.allclose(vals["intrinsic"] + vals["extrinsic"], vals["mid"]))
c.assert_true("extrinsic far below spot is near zero",
              lambda: abs(row(50)["extrinsic"]) < 0.5)
c.assert_true("extrinsic peaks at the at-the-money strike",
              lambda: m.peak(vals)["strike"] == 100)

stale = toy.assign(lastTradeDate=["2026-03-12", "2026-10-23", "2026-10-23",
                                  "2026-10-22", "2026-10-19", "2026-10-23"])
c.assert_true("contracts with no trade in a week are dropped when asof is given",
              lambda: 50 not in set(m.split_call_values(stale, SPOT, asof="2026-10-25")["strike"]))
c.assert_true("recently traded contracts are kept",
              lambda: {90, 100, 110, 150} <= set(m.split_call_values(stale, SPOT,
                                                                     asof="2026-10-25")["strike"]))

near = m.nearest_strikes(pd.DataFrame({"strike": np.arange(80.0, 121.0)}), 100.3, n=10)
c.assert_true("nearest_strikes returns ten rows around spot, in order",
              lambda: list(near["strike"]) == list(np.arange(96.0, 106.0)))

# ---------------------------------------------------------------- Q6
longer = vals.assign(extrinsic=vals["extrinsic"] + 2.0)
c.assert_true("above_everywhere reports 100% when one curve dominates",
              lambda: m.above_everywhere(vals, longer)[0] == 1.0)

sys.exit(c.report())
