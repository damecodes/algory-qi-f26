"""Homework 6 self-check.   python check_hw06.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _check import Checker
import numpy as np

c = Checker("Homework 6 self-check", "hw06_starter.py")
m = c.load(__file__)


def _capped(m):
    lo = m.covered_call(100.0, 105.0, 4.0, 120.0)
    hi = m.covered_call(100.0, 105.0, 4.0, 200.0)
    return abs(lo - hi) < 1e-9


from math import exp

S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.04, 0.25

c.check("Black-Scholes call, the reference case",
        lambda: m.black_scholes("call", S, K, T, R, SIG), 11.84, tol=0.02,
        note="if far off, check that time is in years and rate/vol are decimals")
c.check("Black-Scholes put, the reference case",
        lambda: m.black_scholes("put", S, K, T, R, SIG), 7.92, tol=0.02,
        note="same inputs, the other side")

c.assert_true("put-call parity holds for your pricer",
              lambda: abs((m.black_scholes("call", S, K, T, R, SIG)
                           - m.black_scholes("put", S, K, T, R, SIG))
                          - (S - K * exp(-R * T))) < 0.01,
              note="C - P must equal S - K*exp(-rT). If it does not, one of the two is wrong.")
c.assert_true("a deep in-the-money call is worth about its intrinsic value",
              lambda: abs(m.black_scholes("call", 200, 100, 0.01, R, SIG) - (200 - 100 * exp(-R * 0.01))) < 0.5,
              note="with almost no time left there is almost no extrinsic value")
c.assert_true("price rises with volatility, for both calls and puts",
              lambda: (m.black_scholes("call", S, K, T, R, 0.40) > m.black_scholes("call", S, K, T, R, 0.10)
                       and m.black_scholes("put", S, K, T, R, 0.40) > m.black_scholes("put", S, K, T, R, 0.10)),
              note="higher volatility helps the option owner either way")
c.assert_true("price rises with time to expiry",
              lambda: m.black_scholes("call", S, K, 2.0, R, SIG) > m.black_scholes("call", S, K, 0.5, R, SIG),
              note="more time is more chance to finish in the money")

c.check("implied_volatility recovers the volatility you started from",
        lambda: m.implied_volatility(m.black_scholes("call", S, K, T, R, SIG), S, K, T, R, "call"),
        SIG, tol=0.0005,
        note="feed your own price back in and you must get 0.25 out")
c.check("implied_volatility works on puts too",
        lambda: m.implied_volatility(m.black_scholes("put", S, K, T, R, 0.40), S, K, T, R, "put"),
        0.40, tol=0.0005, note="the same solver, the other contract type")
c.check("implied_volatility at a different strike",
        lambda: m.implied_volatility(m.black_scholes("call", S, 120, T, R, 0.18), S, 120, T, R, "call"),
        0.18, tol=0.0005, note="out of the money should still invert cleanly")

c.assert_true("a covered call caps its own upside",
              lambda: _capped(m),
              note="long stock plus a short call cannot keep gaining above the strike")


sys.exit(c.report())
