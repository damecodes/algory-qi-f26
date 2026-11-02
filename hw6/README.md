# HW6: Greeks, combinations, and implied volatility

This folder holds:

- a Black-Scholes pricer and a bisection implied-volatility solver;
- four option combinations built from live SPY quotes;
- the SPY implied-volatility smile, compared with realised volatility;
- a 2005–2026 simulation of selling weekly at-the-money straddles.

## Result

Run on 2026-11-01, using quotes from the Friday 2026-10-30 close. SPY spot was $755.72 and the expiry used was 2026-11-30.

- **Reference prices:** Black-Scholes gives 11.84 for the call and 7.92 for the put, and put-call parity holds to 4 decimal places.
- **Implied vs realised:** at-the-money implied vol is **15.97%**, the VIX is **19.42**, and 30-day realised vol is **11.32%**.
- **The smile:** implied vol is 29% at 10% below spot and 12% at 5% above spot. This is the equity-index skew.
- **Selling weekly straddles:**
  - it wins 71% of weeks, but loses 15% of notional in a single week in October 2008;
  - an account selling 8× its equity in notional is wiped out.

See [writeup.md](writeup.md) for the full answers.

## Run it

Pull live data from yfinance and save it to `data/`:

```bash
python hw06_starter.py
```

Rerun on the saved data instead:

```bash
python hw06_starter.py --snapshot
```

Run the 10 self-checks:

```bash
python check_hw06.py
```

You need `numpy`, `pandas`, `matplotlib` and `yfinance`. `_check.py` is the same stand-in helper as in HW4.

## Files

| File | What it is |
|------|------------|
| `hw06_starter.py` | the pricer, the IV solver, the combinations, and the output for Q1–Q7 |
| `check_hw06.py`, `_check.py` | self-checks (10/10 pass) |
| `hw06_output.txt` | printed output of the run |
| `data/` | the saved chain, T-bill rate, VIX, SPY closes, and SPY/VIX history since 2005 |
| `writeup.md`, `HW6_writeup.pdf` | the answers |
| `q1_combinations.png`, `q2_calendar_spread.png`, `q5_iv_smile.png`, `q7_short_straddle.png` | plots |
