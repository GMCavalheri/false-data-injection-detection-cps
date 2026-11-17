"""Telling a broken sensor from a lying one.

``fdicps`` simulates a synthetic GNSS/INS navigation chain (two embedded GNSS/INS units and a
barometric altitude source feeding a mission computer over a data bus), injects two kinds of
anomaly into its telemetry, genuine physical sensor faults and false data injection (FDI) at
increasing levels of sophistication, and studies how well each kind can be detected and told
apart: a naive anomaly detector, a physics-informed classifier, a model-based consistency check
(weighted least squares and a Kalman filter) and an honest robustness analysis.
"""

__version__ = "1.0.0"
