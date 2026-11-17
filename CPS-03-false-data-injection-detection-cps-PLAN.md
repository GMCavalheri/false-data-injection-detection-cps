# PLAN.md — CPS-03: False Data Injection Detection in Cyber-Physical Systems

## Overview
A machine learning project extending the fault-detection methodology already built in the aviation reliability portfolio (Fault Detection in Engine Parts, ML-Based Structural Health Monitoring) into an adversarial setting: distinguishing genuine physical sensor failure from malicious False Data Injection (FDI) attacks on a cyber-physical system's sensor telemetry.

## Objective
Build an ML detection pipeline that classifies anomalous sensor readings into "genuine physical fault" vs. "likely false data injection," using the statistical and physical signatures that distinguish the two — a distinction that matters enormously in practice, since the correct response to each is completely different (physical fault → maintenance action; FDI → security incident response).

## Why This Project
Every prior fault-detection project in the portfolio implicitly assumes anomalous sensor data reflects a real physical problem. In an adversarial setting, that assumption fails: FDI attacks (well-documented in the SCADA/ICS security research literature, particularly for power grid state estimation) deliberately craft sensor readings that pass basic sanity/consistency checks while still corrupting the system's understanding of its own state. Being able to tell the difference between "the sensor is broken" and "someone is lying to the sensor's downstream consumer" is a distinctly cyber-physical reliability skill that neither pure reliability engineers nor pure cybersecurity analysts typically build on their own.

## Knowledge Covered
1. **Physics-Based Data Generation with Dual Anomaly Sources**: extend the physics-based simulation approach from prior fault-detection projects to generate two distinct anomaly classes over the same sensor network — genuine physical fault signatures (drift, sudden step change consistent with real component failure physics) and synthetic FDI signatures (crafted to satisfy basic consistency constraints while still being false, following patterns described in the public FDI research literature).
2. **Baseline Anomaly Detection (Undifferentiated)**: implement a standard anomaly detector (the kind used in the prior portfolio projects) that only flags "anomalous vs. normal" without distinguishing cause — establish this as the naive baseline and demonstrate its limitation (it cannot tell a technician whether to send a mechanic or a security analyst).
3. **Physical-vs-FDI Classification**: train a classifier to distinguish the two anomaly classes using features informed by the physical plausibility of the anomaly's temporal/spatial signature (e.g., a real physical fault typically propagates consistently with known physics across correlated sensors; a naive FDI attack often does not, unless the adversary explicitly modeled that correlation).
4. **Adversarial Robustness Consideration**: analyze how detection performance degrades as the simulated FDI attacks become more sophisticated (i.e., increasingly consistent with real physical correlation structure), producing an honest assessment of the detector's limits rather than an overstated result.
5. **Cross-Sensor Consistency Modeling**: implement a model-based consistency check (e.g., a simplified state estimator or the physics models from earlier portfolio projects) as an additional detection feature, reflecting how real power-grid/ICS FDI detection research approaches the problem.

## Prerequisites / Background
- The fault-detection methodology from Fault Detection in Engine Parts and ML-Based Structural Health Monitoring
- Basic understanding of FDI attacks: an adversary manipulates sensor readings (or the data path) to inject false values, typically crafted to be individually plausible and to satisfy simple state-estimation consistency checks
- Classical anomaly detection concepts

## Data / Dataset
- **Synthetic, physics-generated data (primary)**: sensor telemetry from simulation models reused from prior fault-detection projects, with two injected anomaly classes — genuine physical faults (drift, bias, stuck, noise increase) and false data injection (crafted, individually plausible values).
- **Ground truth by construction**: every anomaly's class, start time, and magnitude are known.
- **Optional public ICS security datasets** (e.g., HAI, SWaT), used only under their access terms, to test transfer to real testbed data.

## Tech Stack
- Python, NumPy, Pandas
- Scikit-learn (classical baseline), PyTorch (sequence models, reusing architecture patterns from prior fault-detection projects)
- Reuses physics-based data generators from Fault Detection in Engine Parts / ML-Based Structural Health Monitoring where applicable
- Matplotlib/Plotly for anomaly signature visualization, classification performance by attack sophistication

## Repository Structure
```
false-data-injection-detection-cps/
  src/
    dual_anomaly_data_generation/
    baseline_anomaly_detection/
    physical_vs_fdi_classification/
    adversarial_robustness_analysis/
    cross_sensor_consistency_model/
  external/                # reused physics-based generators from prior projects
  docs/                    # sourcing notes: public FDI research literature only
  notebooks/
  tests/
```

## Execution Plan
- [x] **Phase 1 — Dual anomaly data generation**: extend physics-based simulation to generate both genuine fault and synthetic FDI anomaly classes.
- [x] **Phase 2 — Baseline anomaly detection**: implement and evaluate the naive undifferentiated anomaly detector, document its inability to distinguish cause.
- [x] **Phase 3 — Physical-vs-FDI classification**: implement and train the classifier distinguishing the two anomaly classes.
- [x] **Phase 4 — Adversarial robustness analysis**: systematically increase FDI sophistication, measure and document detection performance degradation.
- [ ] **Phase 5 — Cross-sensor consistency modeling**: implement the model-based consistency check as an additional feature/detection layer.
- [ ] **Phase 6 — Documentation and publishing**: written explanation of the physical-vs-cyber distinction problem, with clear sourcing to public FDI detection research literature (e.g., power grid state estimation FDI research), visualizations of detection performance.

## Validation Strategy
- Baseline and classifier evaluated with standard metrics (precision/recall per class) on held-out synthetic scenarios not used in training.
- Adversarial robustness results reported honestly across a sophistication spectrum, including the point at which detection performance degrades substantially — an honest limitation, not something to hide.

## Deliverables
- Public repository with a documented physical-fault-vs-FDI classification pipeline
- A report on detection performance across FDI sophistication levels, with honest discussion of limits
- Visualizations: anomaly signatures for each class, classification performance curves

## Portfolio Differentiators
- Directly extends existing ML fault-detection expertise into an adversarial setting — a natural, credible bridge for someone with your exact background (reliability ML + growing security interest), rather than a generic "add cybersecurity" bolt-on.
- The honest adversarial robustness analysis (showing where detection breaks down, not just where it works) signals mature judgment valued highly in security-adjacent engineering roles.

## Possible Extensions
- Extend the FDI attack scenarios to specifically target the avionics data bus context established in the Avionics Electronics Reliability project
- Explore using the unified RBD/attack tree framework's output as an additional prior/feature for the classifier (informed by which sensors are more "attractive" attack targets)
