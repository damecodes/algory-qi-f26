"""Homework 7 starter — Algory QI Education, Fall 2026

Machine learning on next-day SPY returns, and an honest account of why it fails.

    python hw07_starter.py              # download 15 years of SPY, save it to data/
    python hw07_starter.py --snapshot   # rerun on the saved data/ files

The check that matters here is that your features never read the future. The
checker builds your features twice, changing only data after row 400, and
confirms the earlier rows did not move.
"""
import importlib.util
import json
import pathlib
import sys
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
HW4_FILE = HERE.parent / "hw4" / "hw04_starter.py"

TICKER = "SPY"
YEARS = 15
TRADING_DAYS = 252
LOOKBACKS = (1, 5, 20, 60)     # past returns, in trading days
VOL_WINDOW = 20                # realised volatility
MA_WINDOW = 50                 # price relative to its moving average
VOLUME_WINDOW = 20             # volume relative to its own average
SEED = 0
RF_TREES = 300
HAND_CHECK_DATE = "2020-03-16"   # the row checked by hand in Q1

# Grids fixed before the first run. Each runs from "memorise the training set"
# to "predict nearly the same number every day", so the curve shows both ends.
RIDGE_ALPHAS = np.logspace(-2, 5, 29)
RF_MIN_LEAF = [1, 2, 5, 10, 20, 50, 100, 200, 400, 800]
GB_MAX_TREES = 500

# Selection rule, fixed before any model was fitted: the final model is the one
# with the highest validation R-squared. The test set is used once, in Q5 and Q6,
# and nothing is changed after seeing it.
RUN_TEST = True

# chart colours: text, secondary text, grid, series 1, series 2
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


# ---------------------------------------------------------------- Q1: features
def feature_names(lookbacks=LOOKBACKS, with_volume=False):
    names = [f"ret_{k}d" for k in lookbacks] + [f"vol_{VOL_WINDOW}d", f"px_vs_ma{MA_WINDOW}"]
    if with_volume:
        names.append(f"volume_vs_{VOLUME_WINDOW}d")
    return names


def _features_from_history(closes, volumes, lookbacks):
    """Features for ONE day, computed from the history that ends with that day.

    `closes` (and `volumes`) end with today's value. This function cannot read
    the future because it is never handed it: build_features passes p[:i+1].
    A feature whose window is not full yet is nan.
    """
    today = closes[-1]
    n = len(closes)
    feats = [today / closes[-1 - k] - 1 if n > k else np.nan for k in lookbacks]

    if n > VOL_WINDOW:                          # 21 closes -> 20 returns ending today
        window = closes[-VOL_WINDOW - 1:]
        daily = window[1:] / window[:-1] - 1
        feats.append(float(np.std(daily, ddof=1) * np.sqrt(TRADING_DAYS)))
    else:
        feats.append(np.nan)

    feats.append(today / np.mean(closes[-MA_WINDOW:]) - 1 if n >= MA_WINDOW else np.nan)

    if volumes is not None:                     # today's volume vs the previous 20 days
        feats.append(volumes[-1] / np.mean(volumes[-VOLUME_WINDOW - 1:-1])
                     if n > VOLUME_WINDOW else np.nan)
    return feats


def build_features(prices, lookbacks=LOOKBACKS, volume=None):
    """One row per trading day: features first, label last.

    Every feature must use only prices up to and including that day. The label is
    the NEXT day's return, and it is the only forward-looking value allowed.

    Row i lines up with prices[i]. Rows whose windows are not full yet (the first
    60) carry nan features, and the last row carries a nan label because its
    next day has not happened. The caller drops those rows.

    Features, from closes (and volume if given):
        ret_1d, ret_5d, ret_20d, ret_60d   return over the past k days, to today's close
        vol_20d                            stdev of the last 20 daily returns, annualised
        px_vs_ma50                         today's close / mean of the last 50 closes - 1
        volume_vs_20d                      today's volume / mean volume of the previous 20 days

    Return a list of rows, each [f1, f2, ..., fn, label].
    """
    p = np.asarray(prices, dtype=float)
    v = None if volume is None else np.asarray(volume, dtype=float)
    rows = []
    for i in range(len(p)):
        history = p[:i + 1]                                    # all that is known at close i
        vol_history = None if v is None else v[:i + 1]
        label = p[i + 1] / p[i] - 1 if i + 1 < len(p) else np.nan   # tomorrow: the label only
        rows.append(_features_from_history(history, vol_history, lookbacks) + [label])
    return rows


# ---------------------------------------------------------------- Q2: split
def split_by_time(rows, train=0.6, val=0.2):
    """Split in time order into (training, validation, test). Never shuffle.

    Shuffling lets 2024 train the model that predicts 2016. It produces
    spectacular, meaningless results and it is the most common mistake in the field.

    Works on a list, an array, or a DataFrame (by position). Every row lands in
    exactly one set, and each set is a contiguous block of time.
    """
    n = len(rows)
    a = int(n * train)
    b = int(n * (train + val))
    take = rows.iloc if hasattr(rows, "iloc") else rows
    return take[:a], take[a:b], take[b:]


def up_day_baseline(prices):
    """Share of days with a positive return: the number any classifier must beat.

    It is nearer 54% than 50%, which is why raw accuracy is a weak measure here.
    """
    p = np.asarray(prices, dtype=float)
    returns = p[1:] / p[:-1] - 1
    return float(np.mean(returns > 0))


# ---------------------------------------------------------------- data
def download_spy(years=YEARS):
    """Daily SPY closes and volume for the last `years` years.

    auto_adjust=True gives closes adjusted for dividends, so a return computed
    from them is what a holder actually earned. When SPY pays a dividend,
    yfinance rescales every earlier close by the same factor. Every feature here
    is a ratio of closes, so that rescaling cancels: it cannot leak anything.
    """
    end = pd.Timestamp(date.today())
    start = end - pd.DateOffset(years=years)
    raw = yf.download(TICKER, start=start.strftime("%Y-%m-%d"),
                      end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(TICKER, axis=1, level="Ticker")
    df = raw[["Close", "Volume"]].rename(columns=str.lower).dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df


def download_tbill(start, end):
    """13-week T-bill yield, ^IRX, in percent a year: what cash could have earned.

    Only used in Q6 to ask what changes if cash pays interest. HW4's backtester
    pays 0% on cash, which flatters buy-and-hold when rates are 4-5%.
    """
    raw = yf.download("^IRX", start=str(start.date()),
                      end=str((end + pd.Timedelta(days=1)).date()),
                      auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "date"
    return raw.rename("irx")


def save_snapshot(spy, irx):
    DATA.mkdir(exist_ok=True)
    spy.to_csv(DATA / "SPY_daily.csv")
    irx.to_csv(DATA / "IRX_daily.csv")
    meta = {"run_date": str(date.today()), "ticker": TICKER, "years": YEARS,
            "first": str(spy.index[0].date()), "last": str(spy.index[-1].date()),
            "rows": len(spy), "source": "yfinance download, auto_adjust=True",
            "tbill": "^IRX close, percent a year"}
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_snapshot():
    spy = pd.read_csv(DATA / "SPY_daily.csv", index_col="date", parse_dates=True)
    irx = pd.read_csv(DATA / "IRX_daily.csv", index_col="date", parse_dates=True)["irx"]
    meta = json.loads((DATA / "meta.json").read_text())
    return spy, irx, meta


def make_table(spy):
    """The feature table: one row per trading day with every window full and a label."""
    names = feature_names(with_volume=True)
    rows = build_features(spy["close"].to_numpy(), volume=spy["volume"].to_numpy())
    table = pd.DataFrame(rows, index=spy.index, columns=names + ["next_ret"])
    return table.dropna(), names


# ---------------------------------------------------------------- Q1 checks
def hand_check(spy, table, names, day):
    """Recompute one row straight from the raw prices, by date, with pandas.

    A second, independent route to every number, which also prints the dates each
    one was built from. Every feature date must be on or before the row's date.
    """
    day = pd.Timestamp(day)
    past = spy.loc[:day]                      # .loc on dates is inclusive: up to that close
    c, v = past["close"], past["volume"]
    row = table.loc[day]
    checks = []
    for k in LOOKBACKS:
        then = c.index[-1 - k]
        checks.append((f"ret_{k}d", c.iloc[-1] / c.iloc[-1 - k] - 1, then,
                       f"close {then.date()} {c.iloc[-1 - k]:8.2f} -> {day.date()} {c.iloc[-1]:8.2f}"))
    daily = c.iloc[-VOL_WINDOW - 1:].pct_change().dropna()
    checks.append((f"vol_{VOL_WINDOW}d", daily.std() * np.sqrt(TRADING_DAYS), daily.index[0],
                   f"{len(daily)} daily returns, {daily.index[0].date()} to {daily.index[-1].date()}"))
    last_ma = c.iloc[-MA_WINDOW:]
    checks.append((f"px_vs_ma{MA_WINDOW}", c.iloc[-1] / last_ma.mean() - 1, last_ma.index[0],
                   f"{len(last_ma)} closes, {last_ma.index[0].date()} to {last_ma.index[-1].date()}"))
    prev_v = v.iloc[-VOLUME_WINDOW - 1:-1]
    checks.append((f"volume_vs_{VOLUME_WINDOW}d", v.iloc[-1] / prev_v.mean(), prev_v.index[0],
                   f"volume {day.date()} / mean of {prev_v.index[0].date()} to {prev_v.index[-1].date()}"))

    nxt = spy.index[spy.index.get_loc(day) + 1]
    label_hand = spy["close"].loc[nxt] / c.iloc[-1] - 1

    print(f"\n    hand check of the row dated {day.date()}, recomputed from raw prices with pandas")
    print(f"    {'column':<15}{'table':>11}{'by hand':>11}   built from")
    worst = 0.0
    for name, hand, _, used in checks:
        worst = max(worst, abs(row[name] - hand))
        print(f"    {name:<15}{row[name]:>+11.5f}{hand:>+11.5f}   {used}")
    print(f"    {'next_ret':<15}{row['next_ret']:>+11.5f}{label_hand:>+11.5f}   "
          f"close {day.date()} -> {nxt.date()}   <- the label, the only value from after {day.date()}")
    newest = max(v.index[-1], c.index[-1])
    print(f"    largest feature difference {worst:.1e}; newest data used by any feature: "
          f"{newest.date()} (the row's own date)")
    return worst


def future_rewrite_check(spy, table, names, day):
    """The checker's test, on the real data: rewrite every close after `day` and rebuild.

    Every feature row up to `day` must be unchanged. The label on `day` must
    change, because it is tomorrow's return and tomorrow was rewritten.
    """
    day = pd.Timestamp(day)
    pos = spy.index.get_loc(day)
    closes = spy["close"].to_numpy(dtype=float).copy()
    volumes = spy["volume"].to_numpy(dtype=float).copy()
    closes[pos + 1:] *= 2.5
    volumes[pos + 1:] *= 3.0
    rows = build_features(closes, volume=volumes)
    redo = pd.DataFrame(rows, index=spy.index, columns=names + ["next_ret"]).loc[table.index]
    upto = table.index <= day
    same = np.allclose(table.loc[upto, names], redo.loc[upto, names], rtol=0, atol=1e-12)
    label_moved = not np.isclose(table.loc[day, "next_ret"], redo.loc[day, "next_ret"])
    print(f"    rewrite every close and volume after {day.date()} and rebuild: "
          f"{upto.sum()} feature rows up to that date unchanged: {same}; "
          f"its label changed: {label_moved}")
    return same and label_moved


# ---------------------------------------------------------------- Q3, Q4: models
def base_models():
    """The four Q3 models at their default settings.

    The linear models sit behind a StandardScaler inside a pipeline, so the scaler
    learns its means and standard deviations from the training rows only. Fitting
    it on the whole table would leak the future's average into the past.
    """
    return {
        "linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge (alpha=1)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random forest": RandomForestRegressor(n_estimators=RF_TREES, random_state=SEED,
                                               n_jobs=-1),
        "gradient boosting": GradientBoostingRegressor(random_state=SEED),
    }


def make_ridge(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def make_forest(min_leaf):
    return RandomForestRegressor(n_estimators=RF_TREES, min_samples_leaf=min_leaf,
                                 random_state=SEED, n_jobs=-1)


def make_boost(n_trees):
    return GradientBoostingRegressor(n_estimators=n_trees, random_state=SEED)


def tune(make, grid, Xtr, ytr, Xva, yva):
    """Fit on training at every grid value, score on validation, keep the best."""
    scores = [r2_score(yva, make(g).fit(Xtr, ytr).predict(Xva)) for g in grid]
    best = int(np.argmax(scores))
    return grid[best], scores


# ---------------------------------------------------------------- Q6: HW4 backtester
def load_hw4():
    """Import the Homework 4 module so its backtest() is used exactly as written."""
    spec = importlib.util.spec_from_file_location("hw04_starter", HW4_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def signal_strategy(decisions, closes):
    """Turn the model's daily hold/cash decisions into a HW4 strategy.

    HW4's backtest calls the strategy once per day, in order. With lookback=1 the
    k-th call is day k and the window is [close k], so the k-th decision is
    returned. That decision came from day k's features (data to close k), and the
    backtester then earns close[k+1] / close[k] - 1: exactly the label the model
    was trained to predict. The window's close is checked against day k's close,
    so a misalignment raises instead of silently shifting every trade by a day.
    """
    state = {"k": 0}

    def strategy(window):
        k = state["k"]
        state["k"] += 1
        if not np.isclose(window[-1], closes[k], rtol=0, atol=1e-9):
            raise AssertionError(f"backtester is on a different day from decision {k}")
        return bool(decisions[k])

    return strategy


def risk_stats(values):
    daily = values[1:] / values[:-1] - 1
    vol = daily.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return vol, sharpe


# ---------------------------------------------------------------- plots
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_tuning(curves, filename):
    """Validation R-squared against each model's hyperparameter, chosen value marked."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (title, xlabel, grid, scores, chosen) in zip(axes, curves):
        style(ax)
        ax.plot(grid, scores, color=BLUE, linewidth=2, marker="o", markersize=4)
        ax.axhline(0, color=INK_2, linewidth=1, linestyle="--")
        i = list(grid).index(chosen)
        ax.plot([chosen], [scores[i]], marker="o", markersize=10, color=BLUE,
                markeredgecolor="white", markeredgewidth=2, zorder=5)
        ax.annotate(f"chosen {chosen:.3g}\nR² {scores[i]:+.4f}", (chosen, scores[i]),
                    textcoords="offset points", xytext=(8, -34), fontsize=9, color=INK,
                    bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none"))
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
    axes[0].set_ylabel("validation R²", fontsize=9, color=INK_2)
    fig.suptitle("Q4  Validation R² against one hyperparameter per model "
                 "(dashed line: R² = 0, as good as guessing the validation mean)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / filename, dpi=150)
    plt.close(fig)


def plot_backtest(dates, strat_values, hold_values, label, filename):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    style(ax)
    ax.plot(dates, hold_values, color=ORANGE, linewidth=2, label="buy and hold SPY")
    ax.plot(dates, strat_values, color=BLUE, linewidth=2, label=label)
    for values, colour in [(hold_values, ORANGE), (strat_values, BLUE)]:
        ax.annotate(f"{values[-1]:.2f}", (dates[-1], values[-1]), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=9, color=INK)
    ax.set_ylabel("value of $1", fontsize=9, color=INK_2)
    ax.set_title("Q6  Test set, out of sample: model rule vs buy and hold",
                 fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / filename, dpi=150)
    plt.close(fig)


def plot_importance(names, values, title, xlabel, filename):
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.barh(np.array(names)[order], np.array(values)[order], color=BLUE, height=0.6)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(HERE / filename, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    snapshot = "--snapshot" in sys.argv
    if not snapshot:
        spy = download_spy()
        save_snapshot(spy, download_tbill(spy.index[0], spy.index[-1]))
    # Always run on the saved files, so a live run and a --snapshot run see the same
    # bits. Yahoo's adjusted closes drift in the 7th digit between downloads, and
    # that alone moves the tree models' results (see robustness.py).
    spy, irx, meta = load_snapshot()

    # ------------------------------------------------------------ Q1
    table, names = make_table(spy)
    print(f"Q1  data: {TICKER} daily, dividend-adjusted closes and volume from yfinance "
          f"({'saved snapshot' if snapshot else 'live download'}, run {meta['run_date']})")
    print(f"    raw       {len(spy):>5} trading days  {spy.index[0].date()} to {spy.index[-1].date()}")
    print(f"    table     {len(table):>5} rows          {table.index[0].date()} to "
          f"{table.index[-1].date()}")
    print(f"              (first {spy.index.get_loc(table.index[0])} days are warm-up for the "
          f"60-day return; the last day has no next day, so no label)")
    print(f"    shape     {table.shape}")
    print(f"    columns   {list(table.columns)}")
    print("    first three rows:")
    with pd.option_context("display.width", 200, "display.float_format", "{:+.5f}".format):
        print("    " + table.head(3).to_string().replace("\n", "\n    "))
    hand_check(spy, table, names, HAND_CHECK_DATE)
    future_rewrite_check(spy, table, names, HAND_CHECK_DATE)

    # ------------------------------------------------------------ Q2
    train, val, test = split_by_time(table)
    print("\nQ2  split by time, 60 / 20 / 20, no shuffling")
    for label, part in [("training", train), ("validation", val), ("test", test)]:
        r = part["next_ret"]
        print(f"    {label:<11}{len(part):>5} rows  {part.index[0].date()} to "
              f"{part.index[-1].date()}   up days {np.mean(r > 0):5.1%}   "
              f"mean {r.mean():+.4%}/day   sd {r.std():.3%}/day")
    disjoint = train.index[-1] < val.index[0] and val.index[-1] < test.index[0]
    print(f"    in order and non-overlapping: {disjoint};   "
          f"rows preserved: {len(train) + len(val) + len(test) == len(table)}")
    assert disjoint

    Xtr, ytr = train[names].to_numpy(), train["next_ret"].to_numpy()
    Xva, yva = val[names].to_numpy(), val["next_ret"].to_numpy()
    Xte, yte = test[names].to_numpy(), test["next_ret"].to_numpy()

    # ------------------------------------------------------------ Q3
    print("\nQ3  four models, default settings, fit on training only")
    print(f"    {'model':<22}{'train R²':>10}{'val R²':>10}")
    fitted = {}
    for label, model in base_models().items():
        model.fit(Xtr, ytr)
        fitted[label] = model
        print(f"    {label:<22}{r2_score(ytr, model.predict(Xtr)):>+10.4f}"
              f"{r2_score(yva, model.predict(Xva)):>+10.4f}")

    # ------------------------------------------------------------ Q4
    print("\nQ4  one hyperparameter each, chosen on validation R²")
    alpha, ridge_scores = tune(make_ridge, list(RIDGE_ALPHAS), Xtr, ytr, Xva, yva)
    leaf, rf_scores = tune(make_forest, RF_MIN_LEAF, Xtr, ytr, Xva, yva)
    # gradient boosting: one fit of 500 trees, scored after every tree
    boost_all = make_boost(GB_MAX_TREES).fit(Xtr, ytr)
    gb_scores = [r2_score(yva, p) for p in boost_all.staged_predict(Xva)]
    gb_grid = list(range(1, GB_MAX_TREES + 1))
    n_trees = gb_grid[int(np.argmax(gb_scores))]

    tuned = {
        "linear regression": fitted["linear regression"],
        f"ridge (alpha={alpha:.3g})": make_ridge(alpha).fit(Xtr, ytr),
        f"random forest (min_samples_leaf={leaf})": make_forest(leaf).fit(Xtr, ytr),
        f"gradient boosting (n_estimators={n_trees})": make_boost(n_trees).fit(Xtr, ytr),
    }
    rows_q4 = [
        ("ridge", "alpha", alpha, ridge_scores, RIDGE_ALPHAS),
        ("random forest", "min_samples_leaf", leaf, rf_scores, RF_MIN_LEAF),
        ("gradient boosting", "n_estimators", n_trees, gb_scores, gb_grid),
    ]
    print(f"    {'model':<19}{'parameter':<18}{'grid':<16}{'chosen':>9}{'val R²':>10}"
          f"{'train R²':>10}")
    for (label, param, chosen, scores, grid), model in zip(rows_q4, list(tuned.values())[1:]):
        print(f"    {label:<19}{param:<18}{f'{grid[0]:g} to {grid[-1]:g}':<16}{chosen:>9.3g}"
              f"{max(scores):>+10.4f}{r2_score(ytr, model.predict(Xtr)):>+10.4f}")
    plot_tuning([("ridge", "alpha (log scale)", RIDGE_ALPHAS, ridge_scores, alpha),
                 ("random forest", "min_samples_leaf (log scale)", RF_MIN_LEAF, rf_scores, leaf),
                 ("gradient boosting", "n_estimators (log scale)", gb_grid, gb_scores, n_trees)],
                "q4_tuning_curves.png")

    # ------------------------------------------------------------ Q5: selection
    val_r2 = {label: r2_score(yva, m.predict(Xva)) for label, m in tuned.items()}
    print("\nQ5  candidates by validation R² (the selection rule, fixed in advance)")
    for label in sorted(val_r2, key=val_r2.get, reverse=True):
        print(f"    {label:<44}{val_r2[label]:>+10.4f}")
    best = max(val_r2, key=val_r2.get)
    model = tuned[best]
    print(f"    best: {best}")

    if RUN_TEST:
        pred = model.predict(Xte)
        r2_test = r2_score(yte, pred)
        hist_mean = ytr.mean()
        r2_os = 1 - np.sum((yte - pred) ** 2) / np.sum((yte - hist_mean) ** 2)
        corr = np.corrcoef(pred, yte)[0, 1]
        test_closes = spy["close"].iloc[spy.index.get_loc(test.index[0]):
                                        spy.index.get_loc(test.index[-1]) + 2]
        hits = np.mean(np.sign(pred) == np.sign(yte))
        print("\n    TEST SET, run once")
        print(f"    test R²                                     {r2_test:+.4f}   "
              f"(sklearn: benchmark is the test set's own mean)")
        print(f"    test R² vs the training-mean forecast       {r2_os:+.4f}   "
              f"(benchmark known in advance: {hist_mean:+.4%}/day)")
        print(f"    correlation, predicted vs actual            {corr:+.4f}   "
              f"(one standard error is about {1 / np.sqrt(len(yte)):.3f})")
        rank_corr = pd.Series(pred).rank().corr(pd.Series(yte).rank())
        biggest = np.argsort(-np.abs(yte))[:5]
        keep = np.ones(len(yte), dtype=bool)
        keep[biggest] = False
        print(f"    rank correlation (Spearman)                 {rank_corr:+.4f}   "
              f"(a few huge days cannot dominate it)")
        print(f"    correlation without the 5 largest moves     "
              f"{np.corrcoef(pred[keep], yte[keep])[0, 1]:+.4f}   (largest: "
              f"{yte[biggest[0]]:+.1%} the day after {test.index[biggest[0]].date()}, "
              f"predicted {pred[biggest[0]]:+.2%})")
        print(f"    sign right                                  {hits:6.1%}   "
              f"vs {up_day_baseline(test_closes):.1%} for 'always up'")
        print(f"    predictions: sd {pred.std():.4%}/day vs actual sd {yte.std():.3%}/day; "
              f"{np.mean(pred > 0):.1%} of them positive")

        # -------------------------------------------------------- Q6
        hw4 = load_hw4()
        decisions = pred > 0
        closes = test_closes.to_numpy()
        strat = hw4.backtest(closes, signal_strategy(decisions, closes), lookback=1)
        hold = hw4.backtest(closes, hw4.always_hold, lookback=1)
        assert np.isclose(hold["total_return"], closes[-1] / closes[0] - 1)
        assert np.isclose(strat["total_return"], np.prod(1 + yte * decisions) - 1)
        print(f"\nQ6  trading rule on the test set, HW4 backtester: hold SPY tomorrow if the "
              f"model predicts a positive return, else cash at 0%")
        print(f"    {test_closes.index[0].date()} to {test_closes.index[-1].date()}, "
              f"{strat['days']} trading days")
        print(f"    {'':<16}{'annual':>9}{'max DD':>9}{'invested':>10}{'ann. vol':>10}"
              f"{'Sharpe':>8}{'total':>9}")
        for label, r in [("model rule", strat), ("buy and hold", hold)]:
            vol, sharpe = risk_stats(r["values"])
            print(f"    {label:<16}{r['annualised_return']:>+9.2%}{r['max_drawdown']:>+9.1%}"
                  f"{r['fraction_invested']:>10.1%}{vol:>10.1%}{sharpe:>8.2f}"
                  f"{r['total_return']:>+9.1%}")
        switches = int(np.sum(decisions[1:] != decisions[:-1]))
        # 1 bp per position change, plus the first entry
        net = ((1 + strat["total_return"]) * (1 - 0.0001) ** (switches + 1)) \
            ** (TRADING_DAYS / strat["days"]) - 1
        print(f"    position changes {switches}; at 1 bp each the rule's annual return is "
              f"{net:+.2%}")
        print(f"    mean next-day return on days the rule held: {yte[decisions].mean():+.4%} "
              f"({decisions.sum()} days);  on days it sat out: "
              f"{yte[~decisions].mean() if (~decisions).any() else float('nan'):+.4%} "
              f"({(~decisions).sum()} days)")

        # Was the timing worth anything beyond being invested 69% of the time?
        rng = np.random.default_rng(SEED)
        log_r = np.log1p(yte)
        shuffled = np.array([log_r[rng.permutation(decisions)].sum() for _ in range(20000)])
        beat = np.mean(shuffled < log_r[decisions].sum())
        median_annual = np.exp(np.median(shuffled) * TRADING_DAYS / len(yte)) - 1
        print(f"    random timing at the same {decisions.mean():.1%} exposure (20,000 shuffles "
              f"of the decisions): median annual {median_annual:+.2%}; the rule beats "
              f"{beat:.1%} of them")

        # What if cash had paid the T-bill rate instead of 0%?
        rate = irx.reindex(test.index).ffill().bfill().to_numpy()
        cash = rate / 100 / TRADING_DAYS
        tbill_rule = np.prod(1 + np.where(decisions, yte, cash)) ** (TRADING_DAYS / len(yte)) - 1
        print(f"    with cash at the 13-week T-bill (average {rate.mean():.2f}%): rule "
              f"{tbill_rule:+.2%} vs buy and hold {hold['annualised_return']:+.2%}; cash paid "
              f"{cash.mean():+.4%}/day vs {yte[~decisions].mean():+.4%} on the days sat out")

        # The same rule on validation, for contrast. That set picked the model, so it flatters.
        val_hold = model.predict(Xva) > 0
        val_rule = np.prod(1 + yva * val_hold) ** (TRADING_DAYS / len(yva)) - 1
        val_bh = np.prod(1 + yva) ** (TRADING_DAYS / len(yva)) - 1
        print(f"    for contrast, validation ({val.index[0].date()} to {val.index[-1].date()}, "
              f"used to pick the model): rule {val_rule:+.2%} vs buy and hold {val_bh:+.2%}; "
              f"days sat out averaged {yva[~val_hold].mean():+.4%}")
        plot_backtest(test_closes.index, strat["values"], hold["values"],
                      f"model rule ({best.split(' (')[0]})", "q6_test_backtest.png")
    else:
        print("\n    RUN_TEST is False: the test set has not been touched.")

    # ------------------------------------------------------------ Q7
    print(f"\nQ7  what the models leaned on (best model: {best})")
    lin = tuned["linear regression"][-1].coef_ * 1e4
    ridge_label = next(k for k in tuned if k.startswith("ridge"))
    rf_label = next(k for k in tuned if k.startswith("random forest"))
    gb_label = next(k for k in tuned if k.startswith("gradient boosting"))
    ridge_coef = tuned[ridge_label][-1].coef_ * 1e4
    rf_imp = tuned[rf_label].feature_importances_
    gb_imp = tuned[gb_label].feature_importances_
    # Diagnostic only, after the one test run: nothing is refitted or re-chosen from it.
    X_perm, y_perm, perm_set = (Xte, yte, "test") if RUN_TEST else (Xva, yva, "validation")
    perm = permutation_importance(model, X_perm, y_perm, scoring="r2", n_repeats=30,
                                  random_state=SEED).importances_mean
    if best.startswith(("linear", "ridge")):
        key = lin if best.startswith("linear") else ridge_coef
        order = np.argsort(-np.abs(key))
    else:
        order = np.argsort(-tuned[best].feature_importances_)
    print("    coefficients: change in predicted next-day return, in basis points, for a "
          "one-standard-deviation move in the feature")
    print("    importances: share of the trees' training-error reduction from splits on "
          "that feature")
    print(f"    {perm_set} drop: fall in {perm_set} R² when that feature's column is "
          f"permuted (best model, 30 repeats)")
    print(f"    {'feature':<16}{'OLS coef':>10}{'ridge coef':>12}{'RF import.':>12}"
          f"{'GB import.':>12}{perm_set[:4] + ' drop':>11}")
    for i in order:
        print(f"    {names[i]:<16}{lin[i]:>+10.2f}{ridge_coef[i]:>+12.3f}{rf_imp[i]:>12.3f}"
              f"{gb_imp[i]:>12.3f}{perm[i]:>+11.5f}")
    if RUN_TEST:
        down = test["ret_1d"].to_numpy() < 0
        calm = test["vol_20d"].to_numpy() <= np.quantile(test["vol_20d"], 1 / 3)
        print(f"    what the rule did on the test set: held {decisions[down].mean():.0%} of days "
              f"after a down day vs {decisions[~down].mean():.0%} after an up day; held "
              f"{decisions[calm].mean():.0%} of the calmest third of days (lowest vol_20d) vs "
              f"{decisions[~calm].mean():.0%} of the rest")
    if best.startswith(("linear", "ridge")):
        values = lin if best.startswith("linear") else ridge_coef
        plot_importance(names, values, f"Q7  {best}: standardised coefficients",
                        "bp of predicted next-day return per 1 sd of the feature",
                        "q7_feature_importance.png")
    else:
        plot_importance(names, tuned[best].feature_importances_,
                        f"Q7  {best}: feature importances",
                        "share of training-error reduction", "q7_feature_importance.png")


if __name__ == "__main__":
    main()
