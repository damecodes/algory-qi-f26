"""Homework 7 self-check.   python check_hw07.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _check import Checker
import numpy as np

c = Checker("Homework 7 self-check", "hw07_starter.py")
m = c.load(__file__)


def _features_ignore_future(m):
    """Build features, then rewrite the future and rebuild.

    Rows early in the table must be byte-identical, because nothing that happened
    later was knowable then. If they move, a feature is reading ahead.
    """
    a = m.build_features(list(PRICES))
    future_changed = list(PRICES)
    for i in range(400, len(future_changed)):
        future_changed[i] *= 2.5
    b = m.build_features(future_changed)
    n = min(len(a), len(b), 300)
    fa = np.asarray(a)[:n]
    fb = np.asarray(b)[:n]
    if fa.ndim > 1:
        fa, fb = fa[:, :-1], fb[:, :-1]      # drop the label column
    return bool(np.allclose(np.nan_to_num(fa), np.nan_to_num(fb), atol=1e-9))

def _splits_ok(m):
    tr, va, te = m.split_by_time(list(range(1000)))
    return (len(tr) and len(va) and len(te)
            and max(tr) < min(va) and max(va) < min(te))

def _splits_total(m):
    rows = list(range(1000))
    tr, va, te = m.split_by_time(rows)
    return len(tr) + len(va) + len(te) == len(rows)


rng = np.random.default_rng(5)
PRICES = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, 600)))


c.exists("build_features is defined", "build_features")
c.assert_true("features never read data from the future",
              lambda: _features_ignore_future(m),
              note="the same prices were passed twice, differing only after row 400. "
                   "Rows before that changed, so a feature is looking ahead.")

c.assert_true("the three splits are in time order and do not overlap",
              lambda: _splits_ok(m),
              note="training must end before validation starts, and validation before test. "
                   "Shuffling would let the future train the model that predicts the past.")


c.assert_true("splitting preserves every row exactly once",
              lambda: _splits_total(m),
              note="the three sets should add back up to the whole dataset")


c.assert_true("directional accuracy is reported against the right baseline",
              lambda: 0.4 <= m.up_day_baseline(list(PRICES)) <= 0.7,
              note="the share of up days is the number to beat, and it is nearer 54% than 50%")

sys.exit(c.report())
