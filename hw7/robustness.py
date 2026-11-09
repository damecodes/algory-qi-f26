"""How much of the Q5/Q6 result is the market, and how much is luck of the fit?

    python robustness.py

Keeps everything from hw07_starter.py fixed (the saved data, the split, the
chosen model: a random forest with min_samples_leaf=50) and changes only things
that should not matter:

    A. the forest's random seed (which bootstrap samples and features each tree sees)
    B. one-in-a-million noise on every close, the size of the drift between two
       downloads of the same Yahoo data

If the model had found something real, the test results would barely move.
Nothing here is used to choose or change the model: it was chosen on validation.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import hw07_starter as hw

RUNS = 20
LEAF = 50          # the value Q4 chose on validation


def evaluate(spy, seed):
    table, names = hw.make_table(spy)
    train, val, test = hw.split_by_time(table)
    model = RandomForestRegressor(n_estimators=hw.RF_TREES, min_samples_leaf=LEAF,
                                  random_state=seed, n_jobs=-1)
    model.fit(train[names].to_numpy(), train["next_ret"].to_numpy())
    y = test["next_ret"].to_numpy()
    pred = model.predict(test[names].to_numpy())
    hold = pred > 0
    years = len(y) / hw.TRADING_DAYS
    return {"val R²": r2_score(val["next_ret"], model.predict(val[names].to_numpy())),
            "test R²": r2_score(y, pred),
            "corr": np.corrcoef(pred, y)[0, 1],
            "invested": hold.mean(),
            "annual": np.prod(1 + y * hold) ** (1 / years) - 1,
            "bh": np.prod(1 + y) ** (1 / years) - 1}


def summarise(label, results):
    keys = ["val R²", "test R²", "corr", "invested", "annual"]
    print(f"\n{label}")
    print(f"    {'':<10}" + "".join(f"{k:>11}" for k in keys))
    for name, fn in [("min", np.min), ("median", np.median), ("max", np.max)]:
        cells = []
        for k in keys:
            v = fn([r[k] for r in results])
            cells.append(f"{v:>+11.2%}" if k in ("invested", "annual") else f"{v:>+11.4f}")
        print(f"    {name:<10}" + "".join(cells))
    bh = results[0]["bh"]
    wins = sum(r["annual"] > bh for r in results)
    print(f"    buy and hold {bh:+.2%} a year; the rule beat it in {wins} of {len(results)} runs")


def main():
    spy, _, meta = hw.load_snapshot()
    print(f"Robustness of the Q5/Q6 test result (data saved {meta['run_date']}, "
          f"random forest, min_samples_leaf={LEAF}, {hw.RF_TREES} trees)")

    base = evaluate(spy, hw.SEED)
    print(f"    as reported (seed {hw.SEED}): test R² {base['test R²']:+.4f}, "
          f"rule {base['annual']:+.2%} a year vs buy and hold {base['bh']:+.2%}")

    summarise(f"A. same data, {RUNS} forest seeds",
              [evaluate(spy, seed) for seed in range(RUNS)])

    rng = np.random.default_rng(1)
    jittered = []
    for _ in range(RUNS):
        noisy = spy.copy()
        noisy["close"] = noisy["close"] * (1 + rng.normal(0, 1e-6, len(noisy)))
        jittered.append(evaluate(noisy, hw.SEED))
    summarise(f"B. seed {hw.SEED}, {RUNS} draws of 1e-6 noise on every close", jittered)


if __name__ == "__main__":
    main()
