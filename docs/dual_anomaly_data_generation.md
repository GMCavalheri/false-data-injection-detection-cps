# Theory note: dual anomaly data generation

**Package:** [`fdicps.dual_anomaly_data_generation`](../src/fdicps/dual_anomaly_data_generation/) ·
**Lesson:** [1](lessons/01-two-kinds-of-wrong.md) ·
**Results:** [`generation.md`](../validation/results/generation.md)

## Kinematics

Each axis (along-track $x$, vertical $h$) has state $[p, v]$ driven by an acceleration held constant
over each 1-s sample. The zero-order-hold solution is exact:

```math
\begin{bmatrix} p_{k+1} \\ v_{k+1} \end{bmatrix} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}\begin{bmatrix} p_{k} \\ v_{k} \end{bmatrix} + \begin{bmatrix} \Delta t^{2}/2 \\ \Delta t \end{bmatrix} a_{k}.
```

A segment lasts 300 s; up to three raised-cosine acceleration pulses per axis, each peaking at
no more than 1 m/s², move the velocity between targets (along-track 180 to 250 m/s, vertical
within 15 m/s). `mission.integrate()` is the vectorised recursion.

## Sensors

13 channels at 1 Hz: per embedded GNSS/INS unit (`egi_1`, `egi_2`, the CPS-02 components) a GNSS
position and velocity and an inertial acceleration on each axis, plus a barometric altitude
(`air_data`, an addition to the CPS-02 architecture). White Gaussian noise per channel; a constant
accelerometer bias per unit. Values and bases: [`sensor_parameters.csv`](../src/fdicps/data/sensor_parameters.csv).

## Physical faults (one unit)

| kind | effect from onset $k_{0}$ |
|---|---|
| bias | step $b$ on the unit's positions (GNSS), accelerations (INS) or altitude (baro) |
| drift | ramp $r(k - k_{0})\Delta t$ on the same channels |
| stuck | every output of the unit frozen at sample $k_{0} - 1$ |
| noise | extra noise so the standard deviation becomes $m\sigma$ |
| common mode | both receivers' positions, the same saturating ramp $\min(r\tau, e_{\max})$ |

## False data injection

An additive vector on the measured values; levels differ in the physics they respect:

| level | pattern | consistency $s$ | construction |
|---|---|---|---|
| L0 | replace | n/a | one channel replaced by a noise-free line plus a jump |
| L1 | offset | 0 | offset ramped into one position channel |
| L2 | offset | 1 | the same offset on every position channel of the axis: $a = Hc$, $c = [\Delta p, 0]$ |
| L3 | drag | 1 | drag-off $\delta a$ for $D$ s, integrated with the kinematics into positions and velocities, added to the INS |

The blend $a(s) = sA + (1-s)A_{\text{target}}$ keeps the target channel's injection fixed. Options
`include_baro` and `include_ins` set the attacker's reach; `cps02_leaf()` maps an injection to the
CPS-02 attack-tree leaf whose effect it is.

## Scenarios

`make_scenario(seed, label)` spawns three independent random streams (trajectory, noise, anomaly),
so one seed gives the same clean telemetry whatever anomaly is injected. `generate_dataset()`
cycles normal, fault and FDI (L0 to L3). Training seeds 0 to 2999 and test seeds 1 000 000 to
1 001 499 never overlap (`fdicps.experiment`).
