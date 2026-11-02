# Homework 6: Greeks, combinations, and implied volatility

All numbers come from `hw06_starter.py`, run on 2026-11-01. That was a Sunday, so the quotes are from the **Friday 2026-10-30 close**. The full printed output is in `hw06_output.txt` and the plots are the `q*.png` files. The chain, the rate, the VIX and the price history used are saved in `data/`, so `python hw06_starter.py --snapshot` reproduces every number.

## Setup

| Setting | Value |
|---|---|
| Underlying | SPY, spot $755.72 |
| Expiry | 2026-11-30, 31 days out (the expiry nearest 30 days, to match the VIX horizon) |
| Risk-free rate | 3.973%, the 13-week T-bill (`^IRX`) on 2026-10-30 |
| Option prices | mid of bid and ask |
| Quotes dropped | zero bid, or no trade in the last 7 days (stale weekend quotes, same filter as HW5) |

## Q1. Four combinations, live SPY quotes

Strikes were picked by a fixed rule:

- **At the money:** the strike nearest spot that has both a call and a put (756).
- **Above spot:** the first call strike at least 2% above spot (771).
- **Below spot:** the last put strike at least 2% below spot (740).

All figures are profit per share at expiry; one contract covers 100 shares.

| Position | Legs | Max profit | Max loss | Break-even |
|---|---|---:|---:|---|
| Covered call | long stock at 755.72, short 771 call at 6.97 | +22.25 | −748.75 | 748.75 |
| Protective put | long stock at 755.72, long 740 put at 7.49 | unlimited | −23.21 | 763.21 |
| Long straddle | long 756 call at 15.17, long 756 put at 12.02 | unlimited | −27.19 | 728.82 and 783.18 |
| Bull call spread | long 756 call at 15.17, short 771 call at 6.97 | +6.80 | −8.20 | 764.20 |

- **Covered call:** the maximum loss is the stock going to zero, cushioned by the premium. The upside stops at the strike plus the premium.
- **Protective put:** the loss is capped at the drop to the put strike plus the premium paid.
- **Long straddle:** SPY has to move about 3.6% either way just to break even.
- **Bull call spread:** max profit is the 15-point strike gap minus the 8.20 net debit.

Plot: `q1_combinations.png`.

## Q2. Why a calendar spread has no single expiry diagram

An expiry payoff diagram only works when every leg expires on the same day. In a calendar spread the one-week call expires while the three-month call still has about twelve weeks left. On that day the long leg is worth its option value, not its intrinsic value, and that value depends on volatility and time as well as on where the stock is. The same SPY price can therefore give very different profits.

If SPY does nothing for a week:

- the short call loses nearly all of its time value;
- the long call loses very little.

So the spread makes money. It is betting that SPY stays near the strike in the near term (short gamma, long theta), and that the back month's implied volatility holds or rises (long vega).

To illustrate, I priced the spread with Black-Scholes at the 16% at-the-money implied vol: a net debit of 20.86. With SPY unchanged after one week, the result depends only on where implied vol has moved (`q2_calendar_spread.png`):

| Implied vol after a week | P/L per share |
|---|---:|
| 11% | −1.55 |
| 16% (unchanged) | +5.58 |
| 21% | +12.74 |

## Q3. Black-Scholes

Spot 100, strike 100, one year, 4% rate, 25% volatility:

- **Call:** 11.84
- **Put:** 7.92
- **Put-call parity:** C − P = 3.9211, which matches S − K·e^(−rT) = 3.9211.

## Q4. Implied volatility

The solver uses bisection between 0.0001 and 5.0, stopping when the interval is narrower than 1e-8. If no volatility in that range reproduces the price, it returns nan. Feeding my own prices back in:

| Contract | Vol in | Vol out |
|---|---:|---:|
| ATM call | 0.2500 | 0.2500 |
| ATM put | 0.2500 | 0.2500 |
| 120 call | 0.1800 | 0.1800 |
| 80 put | 0.6000 | 0.6000 |

All 10 checks in `check_hw06.py` pass.

## Q5. The implied-volatility smile

The 2026-11-30 chain had 225 calls:

- 205 had a non-zero bid;
- 171 of those had also traded in the last week;
- all 171 gave a solvable implied vol.

| Measure | Value |
|---|---:|
| Implied vol at the 756 strike (nearest spot) | **15.97%** |
| VIX on 2026-10-30 | **19.42** (it was 17.1 on 2026-09-14) |
| Implied vol 10% below spot (680 strike) | 29.24% |
| Implied vol 5% above spot (794 strike) | 11.99% |

The shape is the **volatility skew**, also called a "smirk" for equity indices. Implied vol rises steeply at low strikes, bottoms out a little above spot, and turns up slightly at the far right. Under Black-Scholes, with normally distributed log-returns, the line would be flat: one volatility for every strike.

What the skew implies: the market does **not** believe returns are normally distributed. It prices large falls as far more likely than a normal distribution allows, so the left tail is fat. Investors also pay up for downside protection. Since 1987, index puts have carried a crash premium.

Two notes on the plot (`q5_iv_smile.png`):

- **Why the VIX is above the ATM vol.** The VIX (19.4) is higher than the 16.0% ATM vol because it is built from the whole strip of SPX options, skew included, so the expensive downside puts pull it up.
- **Why the left side has two lines.** The deep in-the-money calls on the left sit above the out-of-the-money puts at the same strikes. I believe that is measurement error, not the market: those calls have wide bid-ask spreads, I priced them as European with no dividend while SPY pays about 1.1% a year, and their small vega magnifies any price error into a large IV error. The puts, which I plotted for comparison only, are the cleaner read of the left side, and they show the same skew.

## Q6. Realised vs implied

| SPY volatility | Value |
|---|---:|
| Realised: last 30 daily log returns to 2026-10-30, std × √252 | **11.32%** |
| Implied: at the money, 2026-11-30 | **15.97%** |
| Implied minus realised | **+4.64 vol points** |

## Q7. Selling the gap

**It holds for SPY.** Implied (15.97%) is above realised (11.32%) by 4.6 points today. This is not a one-off: since 2005 the VIX was above the volatility SPY actually realised over the following month on **83% of 5,471 days**, by 3.5 points on average.

**Why a seller can reasonably demand the gap.** Selling an option means selling insurance, and the claims arrive at the worst possible time. A short straddle loses most when the market crashes, which is exactly when the seller's other assets are falling, margin requirements jump and liquidity dries up. Losses are capped on the buyer's side and open-ended on the seller's. A rational seller charges more than the expected payout, the way an insurer does. The gap between implied and realised vol is that insurance premium (the variance risk premium), and the skew in Q5 shows buyers pay the most for crash protection.

**What happens to someone who sells options every week to collect it.** I simulated selling a one-week at-the-money SPY straddle every week from 2005 to 2026: 1,098 weeks, each priced with Black-Scholes at that day's VIX, with no trading costs.

- It made money in **71% of weeks**; the median week earned +0.68% of notional.
- The losses come together:
  - The week of 2008-10-03 lost **14.8% of notional**, which is 22 median weeks of income wiped out in five days.
  - March 2020 produced three of the five worst weeks, and February to April 2020 lost 19.1% in total.
  - The week of 2018-01-29 lost 5.8%; it ends on 5 February 2018, the day of the "Volmageddon" spike.

The VIX overstates what a one-week ATM straddle actually sells for, since it includes the skew; today's ATM vol is 3.4 points below it. So I re-priced every straddle 3 vol points below the VIX. That version wins only 61% of weeks and averages +0.145% a week.

That is a modest edge, and how much leverage the seller uses decides everything (`q7_short_straddle.png`):

| Notional sold per $1 of equity | Outcome |
|---|---|
| 1× | ends at 4.4×, worst drawdown −28% |
| 3× | ends at 38×, worst drawdown −66% |
| 5× | ends at 87×, worst drawdown −87% |
| 8× | **wiped out in the week of 2008-10-03** |

**How it ends.** The trade earns a steady income for years, and the steadier it looks, the more leverage its seller feels safe adding. Then one week collects a decade of premiums in one go. A seller with modest leverage takes a drawdown that most investors and their clients would not sit through. A seller with high leverage is margin-called at the bottom, forced to buy back options at the peak of implied vol, and never gets to collect the premium that follows.

This has happened for real:

- **February 2018:** XIV, an exchange-traded note that was short VIX futures, lost nearly all its value in one day and was shut down by its issuer. The LJM Preservation and Growth fund, which sold S&P options, lost most of its capital the same week.
- **November 2018:** OptionSellers.com clients lost more than their entire accounts when natural-gas calls they had sold spiked.

The premium is real, but it is payment for carrying the crash, and a weekly seller does eventually get the crash.

**Caveats on the simulation:** it uses the VIX as a stand-in for one-week ATM implied vol, has no bid-ask costs or early assignment, and sums returns on notional. Those choices flatter the strategy; the leveraged account shows the compounding risk.
