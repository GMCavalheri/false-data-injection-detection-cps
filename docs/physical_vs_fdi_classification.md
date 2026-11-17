# Theory note: physical-vs-FDI classification

**Package:** [`fdicps.physical_vs_fdi_classification`](../src/fdicps/physical_vs_fdi_classification/) ·
**Lesson:** [3](lessons/03-reading-physics-in-an-anomaly.md) ·
**Results:** [`classification.md`](../validation/results/classification.md), [`sequence_model.md`](../validation/results/sequence_model.md)

## Residuals

Each residual tests one relation and is divided by its noise standard deviation:

* receiver kinematics $r_{k} = p_{k+1} - p_{k} - v_{k}\Delta t - \tfrac12\bar a_{k}\Delta t^{2}$;
* velocity against the INS, $q_{k} = \bar v_{k+1} - \bar v_{k} - \bar a_{k}\Delta t$;
* baro against the mean vertical velocity;
* redundant-pair differences; normalised second differences of every channel.

Window sums telescope: an $n$-sample sum has noise variance $f + ng$ (position terms enter twice,
the others $n$ times). Every residual is centred on the mean over a 60-s reference window, which
adds $n^{2}(f + Rg)/R^{2}$, $R = 60$. `window_z()` is then standard normal on healthy data.

## Features and models

52 features: per unit, frozen fraction, smallest and largest windowed second-difference spread;
per pair, largest, final and steepest disagreement; per relation, the largest 10-s and 60-s
statistic; per axis, whether both receivers broke their kinematics at once.

* `PhysicsClassifier`: scikit-learn `HistGradientBoostingClassifier`, 300 iterations, seed 0, no
  early stopping; three classes (normal, fault, FDI).
* `SequenceClassifier`: a 1-D CNN (three dilated convolutions of width 7, dilations 1, 2, 4;
  global max and mean pooling; linear head) on the 27 residual sequences, AdamW, seeded, one CPU
  thread, deterministic algorithms. On one machine a seed reproduces exactly, but training
  amplifies the tiny floating-point differences between CPUs, so results are reported over five
  seeds and only their mean is checked by CI, within 0.1.
