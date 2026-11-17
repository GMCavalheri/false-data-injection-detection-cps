"""The fixed experimental design shared by every phase, the notebooks and the lessons.

Scenarios are identified by their seed; training and test seeds come from disjoint ranges so
no flight segment, noise realisation or anomaly draw is ever seen on both sides of a split.
"""

from __future__ import annotations

TRAIN_SEEDS = range(0, 3000)
TEST_SEEDS = range(1_000_000, 1_001_500)
ROBUSTNESS_SEEDS = range(2_000_000, 2_000_300)
