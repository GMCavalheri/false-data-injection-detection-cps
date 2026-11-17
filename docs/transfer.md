# Theory note: transfer to HAI

**Package:** [`fdicps.transfer`](../src/fdicps/transfer/) ·
**Lesson:** [6](lessons/06-testing-on-someone-elses-plant.md) ·
**Results:** [`hai_transfer.md`](../validation/results/hai_transfer.md)

## Data handling

HAI 21.03 (`train1`, `test1`, `test2`, about 54 MB) is downloaded by
`uv run python -m fdicps.transfer.fetch` from the dataset repository pinned to commit `2a814ce`,
checked against each file's git blob SHA-1, stored under `data/external/hai` (git-ignored), and
never committed. Cite Shin *et al.* (2020) and the repository when using results.

## Checks

* **Static**: standardise the tags that vary in training; keep the principal subspace $V$ that
  explains 99 % of the variance; $\text{SPE} = \lVert (I - VV^{\top})z \rVert^{2}$ (Jackson and Mudholkar, 1979).
* **Dynamic**: ridge least-squares one-step predictor of the principal scores; Mahalanobis norm of
  the innovation.

Both are smoothed by a 10-s trailing average and thresholded at the 99.9th percentile of their
training values (an empirical choice: neither has a trustworthy closed-form law).

## Measures

Per-second ROC AUC, per-event recall (any alarm inside the attack span), false-alarm episodes per
normal hour, and the share of normal seconds in alarm. The dataset's authors recommend eTaPR,
which is not computed here. HAI labels no physical faults, so attribution is not evaluated.
