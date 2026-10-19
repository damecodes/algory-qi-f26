# Homework 4: Build and honestly test a strategy

All numbers come from `hw04_starter.py`. The full printed output is in `hw04_output.txt`, and the plots are the `q*.png` files in this folder. Data is `^SP500TR` daily closes from Yahoo Finance, 1990-01-02 to 2026-10-16.

## Q1. Data and split

The PDF says to split on September 10, 2011. That was a Saturday, so the split uses Friday, September 9, 2011, the last trading day on or before it (the starter file uses the same date). The test set starts 60 trading days before the split, so the 60-day window is already full on the first out-of-sample day. The strategy reads those 60 days as history but never earns a return on them.

| Set   | Days  | Dates                     | Mean price |
|-------|-------|---------------------------|-----------:|
| Train | 5,469 | 1990-01-02 to 2011-09-09  | 1,358.78   |
| Test  | 3,857 | 2011-06-16 to 2026-10-16  | 6,875.45   |

The test set has 3,857 days: 60 warm-up days and 3,797 out-of-sample days. The training set has 5,469 days, which matches the expected "roughly 5,450". Over the full period the index returned **11.08% a year**, inside the expected 10–11% range.

## Q3–Q5. Backtester checks (training set)

| Strategy             | Total return | Annualised |
|----------------------|-------------:|-----------:|
| Always hold          | +432.3%      | +8.10%     |
| Always cash          | +0.0%        | +0.00%     |
| Random 70% (seed 0)  | +361.4%      | +7.38%     |
| 5-day reversal       | +206.7%      | +5.36%     |

- **Always hold** matches buy-and-hold over the same days to within 2.5e-14, so there is no off-by-one error.
- **Always cash** returns exactly 0%.
- **Random 70%** lands between cash and the index, as it should.
- **5-day reversal**, the rule from class, loses to the index by about 2.7% a year.

All 23 checks in `check_hw04.py` pass, including the lookahead probe.

Plots: `q3_always_hold_vs_index.png` and `q4_q5_controls_training.png`.

## Q6. Every variant I tried (training set only)

I stuck to one family of rules, moving-average trend rules, and tried 17 variants. Before running any of them I fixed how the final rule would be chosen: **the variant with the highest training-set annualised return**. The index returned +8.10% a year over the same period.

| Rule                                   | Annual | Invested | Max DD |
|----------------------------------------|-------:|---------:|-------:|
| close > 10-day SMA                     | +0.67% | 58.6% | −55.4% |
| close > 20-day SMA                     | +2.57% | 61.5% | −46.0% |
| close > 30-day SMA                     | +3.84% | 63.3% | −41.4% |
| close > 40-day SMA                     | +2.07% | 64.6% | −50.1% |
| close > 50-day SMA                     | +4.34% | 66.0% | −36.8% |
| close > 60-day SMA                     | +5.72% | 67.4% | −32.3% |
| close > 0.98 × 60-day SMA              | +4.93% | 80.9% | −51.6% |
| close > 1.02 × 60-day SMA              | +2.18% | 43.6% | −33.9% |
| close > 20-span EMA                    | +3.63% | 63.2% | −50.5% |
| close > 40-span EMA                    | +4.94% | 66.0% | −38.4% |
| 5-day SMA > 20-day SMA                 | +2.66% | 61.3% | −43.2% |
| 10-day SMA > 30-day SMA                | +4.41% | 62.7% | −31.8% |
| 10-day SMA > 50-day SMA                | +5.26% | 66.2% | −28.4% |
| 20-day SMA > 60-day SMA                | +4.49% | 67.3% | −41.8% |
| 20-day SMA rising over 10 days         | +4.47% | 63.1% | −35.3% |
| **40-day SMA rising over 20 days**     | **+7.16%** | 67.5% | −37.8% |
| close > 50-day SMA AND 10 > 50-day SMA | +4.29% | 59.8% | −28.9% |

None of the 17 variants beat buy-and-hold on the training data. Most of them did cut the worst drawdown (the index's was −55.3%), but they gave up too much return to do it.

## Q7. Final strategy

**Rule:** hold the index tomorrow if today's 40-day simple moving average is above the 40-day average from 20 trading days ago. Otherwise hold cash.

This rule says "stay in while the medium-term trend is rising." The reason it might work beyond curve-fitting is time-series momentum: trends in broad indices tend to persist for weeks to months, and that effect is documented across many markets and decades. The biggest losses also tend to come during long declines, which a smoothed trend filter can step out of. Comparing two averages, rather than comparing the price to one average, should also cause fewer false switches on single noisy days.

On the training set the rule returned +7.16% a year, with a −37.8% max drawdown and 67.5% of days invested. The index returned +8.10% a year with a −55.3% max drawdown.

## Q8. Test set, run once

| Out-of-sample, 2011-09-09 to 2026-10-16 | Total   | Annualised | Max drawdown | Invested |
|-----------------------------------------|--------:|-----------:|-------------:|---------:|
| Final rule                              | +250.7% | **+8.68%** | −30.4%       | 77.7%    |
| Index                                   | +773.5% | **+15.47%** | −33.8%      | 100.0%   |

I did not change the strategy after seeing these numbers. Plots: `q7_final_rule_training.png` and `q8_final_rule_test.png`.

## Q9. What the rule says today

**HOLD the index.** The last close used was 2026-10-16. The 40-day SMA is 17,261.51 today, against 17,123.77 twenty trading days ago, so the trend is still rising.

## Q10. Write-up

The strategy lost to buy-and-hold on the test set, and by a lot. Out of sample it made 8.68% a year against the index's 15.47%. A dollar grew to about $3.51 with the rule and about $8.74 in the index. The only thing it delivered was a slightly smaller worst drawdown (−30.4% vs −33.8%), and that is far too small to justify giving up almost 7 points of return a year.

The training result had already told me this would probably happen. Not one of my 17 moving-average rules beat the index on training data, and the one I picked was simply the least bad, trailing by about 0.9% a year. Picking the best of 17 rules on the same data also flatters the winner, so its true training-period edge was probably even worse than −0.9%. What surprised me was how much bigger the gap became out of sample: from about −1% a year to about −7% a year.

Looking at the test-set plot, the reason is clear. From 2011 to 2026 the market mostly went up, and its sell-offs (late 2018, March 2020, 2022, April 2025) recovered quickly. A trend rule only exits after the trend has already turned down, so it takes most of the drop. It then only re-enters after the 40-day average turns back up, so it misses the start of the recovery. In a slow bear market like 2000–2002 or 2008, being out for months saves real money. In a fast V-shaped crash, it means selling low and buying back higher. Being out of the market 22% of the time when it is compounding at 15% a year is very expensive.

I would not invest my own money in this rule. The backtest is also kinder to it than reality would be: it ignores trading costs and taxes on each switch, which would make it look worse. The one thing it leaves out in the rule's favour is that cash earns 0% here, whereas T-bills would have paid something. To trust a rule like this, I would want to see that it:

1. beats buy-and-hold on a risk-adjusted basis across several separate out-of-sample periods and markets, not just one;
2. still does so after realistic costs and taxes;
3. works for a range of nearby settings, not just one lucky choice of 40 and 20 days;
4. has an economic reason to work that I can say before looking at the data, plus some reason to expect that reason to still hold.

Here, my rule failed the very first of those tests.

The main thing I learned is that the test set is the only number that counts. The process mattered too: I fixed the selection rule before running anything, printed every variant, and ran the test once, so I can actually believe the −7%. Had I tried more rule families, or gone back after seeing the test set, I could have found a rule with a nice-looking backtest, and it would have meant nothing. For a broad, total-return index, simply holding it was very hard to beat.
