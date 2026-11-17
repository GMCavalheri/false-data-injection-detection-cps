# Theory note: adversarial robustness analysis

**Package:** [`fdicps.adversarial_robustness_analysis`](../src/fdicps/adversarial_robustness_analysis/) ·
**Lesson:** [4](lessons/04-when-the-attacker-knows-the-physics.md) ·
**Results:** [`robustness.md`](../validation/results/robustness.md)

## Closed forms

For the snapshot model $z = Hx + e$, $W = R^{-1}$ and $S = I - H(H^{\top}WH)^{-1}H^{\top}W$, the bad-data statistic
is $J \sim \chi^2_{m-n}(\lambda)$ with $\lambda = a^{\top}WSa$. Since $SA = 0$,

```math
\lambda(s) = (1 - s)^{2}\,\lambda(0).
```

A scenario of $N$ independent tests at level $\alpha = 1 - (1 - F)^{1/N}$ has false-alarm rate exactly $F$, and
detection probability $1 - \prod_{t}(1 - P_{t})$ with each $P_{t}$ a noncentral chi-square tail
(`snapshot_scenario_detection()`).

## Attack sets

* `consistency_sweep(seeds, pattern)`: the same attacks at $s = 0, 0.1, \dots, 1$, paired by seed.
* `near_mimic(seed)`: a fault-sized step on one receiver position, one axis.
* `exact_mimic(seed)`: a receiver-bias fault's exact effect, labelled FDI; its data have the
  fault's distribution, so no classifier can beat the prior (Bayes).
* `spoofer_variant(seed, reach)`: an L3 drag-off by an attacker who reaches the receivers only
  (CPS-02 `gnss_spoof`), the whole bus, or the bus except the air data.

## Measured quantities

Attribution and detection along the dial for classifiers trained on L0–L1 and on L0–L3, the
breakdown point, mimicry outcomes, and the integrity-monitor miss probability that CPS-02 assumed
to be 0.1.
