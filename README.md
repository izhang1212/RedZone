# RedZone

## Overview

**Description:** A NFL win probability simulator that models in game probability using a Brownian bridge framework

**Method:** An empirical drift table built from 4 seasons of play-by-play data provides the base win probability, which is blended with nflfastR's model and refined through a multi-step Brownian bridge Monte Carlo simulation that captures path-dependent game dynamics.

**Goal:** To generate accurate, real-time win probabilities for NFL games and identify mispriced moneyline bets where the model's probability exceeds the sportsbook's implied probability.

## Models

- **Empirical Drift Table:** Historical play-by-play data binned by score differential (possession-count aware), time remaining, field position, and possession — producing a lookup of actual win rates across ~1,000+ game-state combinations with Bayesian smoothing for sparse bins
- **Brownian Bridge Simulation** — Multi-step Monte Carlo with state-dependent dynamics:
    - Score-aware volatility compression (blowouts converge faster than close games)
    - Clutch-time volatility boost for one-score games in Q4
    - Logit-space paths ensuring probabilities stay bounded (0, 1)
    - Bridge pull term that funnels paths toward terminal outcomes as the clock expires

## Example Output
```
============================================================
Metric                       Our Model     nflfastR   Drift Only
------------------------------------------------------------
Brier Score                     0.1867       0.1840       0.1864
Log Loss                        0.5311       0.5234       0.5295
Plays evaluated                  1,648

Calibration:
     Bin  Predicted     Actual    Count    Error
------------------------------------------------
    5.0%      0.028      0.000      210   -0.028
   35.0%      0.349      0.363      171   +0.014
   55.0%      0.550      0.536      235   -0.014
   65.0%      0.650      0.610      287   -0.041
   85.0%      0.846      0.838       68   -0.007
   95.0%      0.966      1.000      139   +0.034

Accuracy by game phase:
Phase              Brier  LogLoss  nflfast    Plays
--------------------------------------------------
Q1                0.2267   0.6437   0.2034      349
Q3                0.2341   0.6390   0.2349      394
Q4 late           0.0334   0.1243   0.0412      113
Final 2min        0.0126   0.0681   0.0212       86
```

## References

* [Brownian Bridge](https://en.wikipedia.org/wiki/Brownian_bridge) — Stochastic process constrained to hit fixed endpoints, used here to model win probability paths that must terminate at 0 (loss) or 1 (win)
* [Monte Carlo Simulation](https://en.wikipedia.org/wiki/Monte_Carlo_method) — Repeated random sampling to estimate the distribution of possible outcomes
