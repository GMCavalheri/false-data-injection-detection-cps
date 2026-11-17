# CLAUDE.md

Telling a broken sensor from a lying one. A synthetic GNSS/INS navigation chain (two embedded
GNSS/INS units, egi_1 and egi_2, plus a barometric altitude source, feeding a mission computer
over a data bus) produces telemetry into which two kinds of anomaly are injected: genuine
physical sensor faults (bias, drift, stuck, noise increase) and false data injection (FDI) at
four levels of sophistication. The project measures how well a naive anomaly detector, a
physics-informed classifier and a model-based consistency check (weighted least squares and a
Kalman filter) detect each kind and tell them apart, and where that breaks down.
Plan: `CPS-03-false-data-injection-detection-cps-PLAN.md` (tick the phase checkboxes as phases
land). The navigation chain is the one in CPS-02 (`GMCavalheri/cyber-physical-rbd-attack-trees`,
package `cpsrbd`) and CPS-01 (`cyberrel`); keep channel names consistent with them.

## Commands
- `uv sync` — install (Python ≥ 3.11; CI runs 3.11, 3.12 and 3.13). PyTorch comes from the
  PyTorch CPU index (see `[tool.uv.sources]`)
- `uv run pytest --cov` — tests with the 90 % coverage gate
- `uv run ruff check .` and `uv run ruff format --check .` — lint/format. Ruff also formats
  the ```python blocks inside Markdown, so run it after editing docs too
- `uv run mypy` — strict type check of `src/`
- `uv run python validation/<script>.py` — regenerate one phase's tables
  (`validation/results/`) and figures (`docs/figures/`); `validation/run_all.py` runs them all
- `uv run python notebooks/build_notebooks.py` — regenerate and execute the notebooks

## Layout
- `src/fdicps/` — single package; subpackages mirror the plan:
  `dual_anomaly_data_generation` (mission profile, sensors, faults, FDI levels, datasets),
  `baseline_anomaly_detection`, `physical_vs_fdi_classification` (features, gradient
  boosting, the PyTorch sequence model), `adversarial_robustness_analysis`,
  `cross_sensor_consistency_model` (WLS + bad-data test, Kalman filter + NIS/CUSUM, IEEE
  14-bus DC model), `transfer` (HAI loader); plus `plotting.py`.
- `src/fdicps/data/` — parameter tables (CSV); every value is labelled an assumption or
  carries its source.
- `tests/` — pytest + Hypothesis. `validation/` — one script per phase, `results/` tables.
- `docs/` — theory note per module, `scope.md`, `related-work.md`, `report.md`;
  `docs/lessons/` is the course (keep its numbers in sync with `validation/results/`).
  `notebooks/` — one executed demo per lesson.

## Conventions
- SI units: metres, seconds, m/s, m/s². Telemetry is sampled at 1 Hz.
- Ground truth is known by construction: every anomaly carries its class (normal, fault,
  FDI), type, FDI level, start time, magnitude and the channels it touches.
- Everything random takes a seeded `numpy.random.Generator`; the PyTorch model is seeded,
  single-threaded and deterministic. Results in `validation/results/` are rounded so that
  `validation/run_all.py` reproduces them byte for byte.
- Train/test splits are by scenario seed, never by window: no scenario contributes to both.
- Exact results (closed forms) are compared with tolerance 1e-10 or tighter; Monte Carlo
  against its own confidence interval.

## Scope (public repo)
- Synthetic data and public methodology only (the FDI state-estimation literature, public
  GNSS/INS references). FDI is modelled as manipulated measurement values reaching the
  estimator; how an attacker would gain that access is out of scope, with no exploit
  detail. See `docs/scope.md`.
- Public ICS testbed data (HAI) is used only under its terms, downloaded by a script, never
  committed.
- Never mention the author's employer or day job in any file, commit message or doc.

## Docs (GitHub-rendered math)
- Display equations go in ```` ```math ```` fences (no Markdown processing inside).
- Inline `$...$` is processed as Markdown first: no `\,` or `\;`, no `\\` row breaks, and no
  `_` right after `}` `)` `]`. `tests/test_docs.py` guards this.

## Testing rules
- Every test asserts against a closed-form result, an independent oracle (CPS-02's `cpsrbd`,
  the normal equations, the discrete algebraic Riccati equation) or a cited published value;
  name the source in a comment.
- Never loosen a tolerance to make a failing test pass without explaining why.
- Properties that must hold for every parameter set (stealthy attacks leave residuals
  unchanged, detectability never rises with sophistication) are Hypothesis tests.

## Workflow
- Commit small logical changes directly to `main`; push after ruff, mypy and pytest pass,
  each run as its own command (no pipes: a pipe hides the exit code). Stage explicit paths.
- Each phase lands as code + tests, then validation results, then its lesson in
  `docs/lessons/` as a separate commit. Confirm CI is green after each push.
