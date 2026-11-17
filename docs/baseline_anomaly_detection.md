# Theory note: baseline anomaly detection

**Package:** [`fdicps.baseline_anomaly_detection`](../src/fdicps/baseline_anomaly_detection/) ·
**Lesson:** [2](lessons/02-the-detector-that-cannot-say-why.md) ·
**Results:** [`baseline.md`](../validation/results/baseline.md)

## Rule-based checks

The second difference $d_{k} = z_{k} - 2z_{k-1} + z_{k-2}$ removes any locally linear trend; with white noise
of variance $\sigma^{2}$ it has variance $6\sigma^{2}$. A rate test flags $|d_{k}| > \kappa\sigma\sqrt{6}$, with false-alarm
probability $2(1 - \Phi(\kappa))$ per test in straight-and-level flight. The envelope check flags
physically impossible values. `rule_score()` returns the largest rate score of a scenario, or
infinity outside the envelope; the default $\kappa = 5$.

## Isolation Forest

47 generic features per scenario, each normalised by its healthy noise level: per channel, the
standard deviation and maximum of the normalised second difference; per redundant comparison
(receiver against receiver, INS against INS, baro against the receivers' mean), the mean, standard
deviation and largest deviation of the normalised difference. The forest (scikit-learn, 300
trees, seed 0) is trained on normal scenarios only; the alarm threshold is the 99th percentile of
their scores. Score $s(x, n) = 2^{-E[h(x)]/c(n)}$ (Liu, Ting and Zhou, 2008), sign flipped so larger
means more anomalous.

## What it is for

It is the competent, cause-blind baseline: one number per scenario. The results show it detects
loud anomalies, misses consistent attacks, and ranks faults above attacks.
