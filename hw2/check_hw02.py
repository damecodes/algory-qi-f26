"""Homework 2 self-check.   python check_hw02.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _check import Checker
import numpy as np

c = Checker("Homework 2 self-check", "hw02_starter.py")
m = c.load(__file__)


def _bond(face, cr, years, r):
    """Reference implementation, so expectations are computed, never typed."""
    cpn = face * cr
    return sum(cpn / (1 + r) ** t for t in range(1, years + 1)) + face / (1 + r) ** years


c.check("present_value, the worked example",
        lambda: m.present_value([10, 15, 20], 0.10), 36.51,
        note="10/1.1 + 15/1.21 + 20/1.331")
c.check("present_value, one flow one year out",
        lambda: m.present_value([50], 0.10), 50 / 1.1, tol=0.001,
        note="the first flow arrives in one year, not today")
c.check("present_value at a zero rate",
        lambda: m.present_value([10, 10, 10], 0.0), 30.0,
        note="with no discounting the sum is unchanged")

c.check("bond_price at its own coupon rate",
        lambda: m.bond_price(1000, 0.04, 10, 0.04), 1000.0, tol=0.005,
        note="must come out at exactly face value")
c.check("bond_price below its coupon rate",
        lambda: m.bond_price(1000, 0.04, 10, 0.02), _bond(1000, 0.04, 10, 0.02), tol=0.02,
        note="rates down, price up")
c.check("bond_price at today's 10-year yield",
        lambda: m.bond_price(1000, 0.04, 10, 0.0496), _bond(1000, 0.04, 10, 0.0496), tol=0.02,
        note="rates up, price down")
c.check("bond_price with a zero coupon",
        lambda: m.bond_price(1000, 0.0, 10, 0.05), _bond(1000, 0.0, 10, 0.05), tol=0.02,
        note="just the discounted face value")

flat = [100.0] * 300
grow = [100.0 * (1.0004 ** i) for i in range(253)]
c.check("annualised_volatility of a series that never moves",
        lambda: m.annualised_volatility(flat), 0.0, tol=1e-6,
        note="no movement means no volatility")
c.check("annualised_return of a steadily growing series",
        lambda: m.annualised_return(grow), 1.0004 ** 252 - 1, tol=0.01,
        note="0.04% a day compounded over a trading year")

rng = np.random.default_rng(0)
mkt = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800)))
lev = 100 * np.exp(np.cumsum(np.diff(np.log(mkt), prepend=np.log(100)) * 2))
indep = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, 800)))

c.check("beta of the market against itself",
        lambda: m.beta(list(mkt), list(mkt)), 1.0, tol=1e-6,
        note="anything else means something is badly wrong")
c.check("beta of a stock with market-independent returns",
        lambda: m.beta(list(indep), list(mkt)), 0.0, tol=0.15,
        note="near zero on returns. A large value here means you used prices, not returns.")
c.check("beta of a 2x levered copy of the market",
        lambda: m.beta(list(lev), list(mkt)), 2.0, tol=0.05,
        note="double the market's moves, double the beta")

sys.exit(c.report())
