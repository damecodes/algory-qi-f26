"""Homework 3 self-check.   python check_hw03.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _check import Checker

c = Checker("Homework 3 self-check", "hw03_starter.py")
m = c.load(__file__)

_EPV_HALF = sum((0.5 ** i) * 10 / 1.1 ** (i + 1) for i in range(3))

c.check("expected_present_value with 50% survival each year",
        lambda: m.expected_present_value([10, 10, 10], 0.10, 0.5), _EPV_HALF, tol=0.02,
        note="year 1 certain, year 2 at 0.5, year 3 at 0.25. This is NOT the 16.98 from "
             "the Session 3 slide, which modelled a single shutdown event rather than a "
             "yearly one. Read the docstring.")
c.check("expected_present_value with certain survival",
        lambda: m.expected_present_value([10, 15, 20], 0.10, 1.0), 36.51, tol=0.02,
        note="survival of 1.0 must reduce to plain present value")
c.check("expected_present_value with no survival past year 1",
        lambda: m.expected_present_value([10, 10, 10], 0.10, 0.0), 10 / 1.1, tol=0.02,
        note="only the first year is ever collected")

c.check("simulated P(4-sided | rolled a 1)",
        lambda: m.simulate_dice(200_000), 0.6, tol=0.02,
        note="exact answer is 3/5. If you are far off, check which die you recorded.")
c.check("simulated three-coin expected payout",
        lambda: m.simulate_coins(200_000), 1.5, tol=0.02,
        note="pays heads x tails, so 0, 2, 2, 0 across the four cases")

c.assert_true("simulate_dice is reproducible with a fixed seed",
              lambda: abs(m.simulate_dice(50_000, seed=1) - m.simulate_dice(50_000, seed=1)) < 1e-12,
              note="the same seed must give the same answer twice, or your results are not reproducible")
c.assert_true("more trials gives a closer estimate",
              lambda: abs(m.simulate_coins(400_000, seed=3) - 1.5)
                      <= abs(m.simulate_coins(400, seed=3) - 1.5) + 0.05,
              note="this is the law of large numbers, and it is the point of question 3")

# ---------------------------------------------------------------- Q5/Q6
# Hand-built return series, so every expected value below can be counted on
# paper. No download, so these run with no network and give the same answer
# every time. UP and DOWN are ordinary days; BIG is past the -2% threshold and
# SMALL is a down day that must NOT count as one.
UP, DOWN, SMALL, BIG = 0.01, -0.01, -0.005, -0.03

ALTERNATING = [UP, DOWN, UP, DOWN, UP, DOWN]
BLOCKS      = [DOWN, DOWN, DOWN, DOWN, UP, UP, UP, UP]
ALL_DOWN    = [DOWN, DOWN, DOWN, DOWN]
MIXED_AFTER = [BIG, UP, BIG, DOWN, UP, SMALL]
BIG_REBOUND = [BIG, UP, DOWN, DOWN, BIG, UP]

c.check("p_down on a series that alternates",
        lambda: m.p_down(ALTERNATING), 0.5, tol=1e-9,
        note="three of the six days are down")
c.check("p_down when every day is down",
        lambda: m.p_down(ALL_DOWN), 1.0, tol=1e-9,
        note="all four days are down, so the answer is 1, not 0")

c.check("p_down_given_down when a down day never follows a down day",
        lambda: m.p_down_given_down(ALTERNATING), 0.0, tol=1e-9,
        note="the series alternates, so no down day is ever followed by another. "
             "Getting 0.5 means you returned the unconditional P(down) instead of "
             "conditioning on the day before.")
c.check("p_down_given_down on four down days then four up",
        lambda: m.p_down_given_down(BLOCKS), 3 / 4, tol=1e-9,
        note="four days are down; three of them are followed by a down day and the "
             "fourth is followed by the first up day")

c.check("p_down_given_big_drop ignores drops above the threshold",
        lambda: m.p_down_given_big_drop(MIXED_AFTER), 0.5, tol=1e-9,
        note="only the two -3% days qualify. One is followed by an up day and one by "
             "a down day. The -1% and -0.5% days are down days but not big drops.")
c.check("p_down_given_big_drop is not p_down_given_down",
        lambda: m.p_down_given_big_drop(BIG_REBOUND), 0.0, tol=1e-9,
        note="in this series every big drop is followed by an up day, while ordinary "
             "down days are followed by down days. p_down_given_down here is 0.5, so "
             "getting 0.5 means the threshold is being ignored.")

sys.exit(c.report())
