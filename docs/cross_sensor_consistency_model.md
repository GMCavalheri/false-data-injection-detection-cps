# Theory note: cross-sensor consistency model

**Package:** [`fdicps.cross_sensor_consistency_model`](../src/fdicps/cross_sensor_consistency_model/) ·
**Lesson:** [5](lessons/05-state-estimation-as-a-lie-detector.md) ·
**Results:** [`consistency.md`](../validation/results/consistency.md)

## Weighted least squares (`wls.py`)

Generic $z = Hx + e$: estimate, residual, $J = r^{\top}Wr$, its threshold, noncentrality and detection
probability. Used for the navigation snapshot check (`navigation.py`, 2 and 3 degrees of freedom)
and for the IEEE 14-bus DC model (`ieee14.py`, 34 meters, 13 angles, 21 degrees of freedom; branch
data from MATPOWER `case14`, susceptance $1/(xt)$ as in `makeBdc`).

## Kalman filter (`kalman.py`)

Per axis, state $[p, v, b]$, input the mean INS acceleration:

```math
F = \begin{bmatrix} 1 & \Delta t & -\Delta t^{2}/2 \\ 0 & 1 & -\Delta t \\ 0 & 0 & 1 \end{bmatrix}, \quad G = \begin{bmatrix} \Delta t^{2}/2 \\ \Delta t \\ 0 \end{bmatrix}, \quad Q = GG^{\top}\frac{\sigma_{a}^{2}}{2} + \operatorname{diag}(0, 0, q_{b}).
```

Covariances and gains are data-independent and computed once (Joseph form); the prior covariance
converges to the DARE solution. NIS is $\chi^2_{m}$ and independent in time when the model holds.

## Monitors (`detection.py`)

Per-sample NIS and 20-s block sums ($\chi^2_{20m}$) with exact scenario false-alarm rates; CUSUM
$g_{k} = \max(0, g_{k-1} + N_{k} - 1.5m)$ with a threshold calibrated on training normals. Fourteen
consistency features summarise them for the classifier.

## Theorem used

If $a_{k} = Hc_{k}$ with $c_{k+1} = Fc_{k} + G\delta u_{k}$, $c_{0} = 0$, and $\delta u_{k}$ is added to the input, every
innovation equals the clean one and the estimate is offset by exactly $c_{k}$. Only a measurement
outside the attacker's reach breaks it, in the directions that measurement observes.
