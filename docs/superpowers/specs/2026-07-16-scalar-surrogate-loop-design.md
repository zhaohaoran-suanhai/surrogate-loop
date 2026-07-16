# Scalar Surrogate Loop Design

## 1. Goal

Build the smallest auditable workflow proving that a natural-language request can safely drive a deterministic surrogate-model training toolchain.

The first supported problem is fixed to

\[
\frac{du}{dt}=\gamma u+0.5t,\qquad
\gamma\in[-1,1],\qquad
t\in[0,1],\qquad
u(0)=0,
\]

and the first prediction contract is

\[
\gamma\longmapsto u(1).
\]

Codex is the natural-language interface in this version. It translates an approved user request into a structured configuration and invokes a fixed Python CLI. The runtime never evaluates user-provided equations or executes generated training code.

## 2. Alternatives Considered

### A. Workflow-first scalar surrogate - selected

Predict only `gamma -> u(1)` and compare a polynomial response surface, Gaussian process regression, and a small multilayer perceptron. This is the smallest design that still tests configuration validation, numerical data generation, leakage-free splitting, model comparison, artifact persistence, reloadable inference, and out-of-domain rejection.

### B. Full-trajectory surrogate

Predict `gamma -> [u(t_1), ..., u(t_101)]`, preferably with PCA/POD followed by a surrogate for the reduced coefficients. This is closer to field or operator learning, but adds output reduction, reconstruction metrics, and trajectory visualization before the workflow itself has been validated.

### C. Neural-operator-first pipeline

Start with DeepONet or an equivalent coordinate model. This is closest to PDEFlow, but it introduces PyTorch, architecture choices, GPU variability, longer training, and more difficult failure diagnosis. It does not improve the evidence needed from the first loop.

The selected approach is A. Approach B is the next scientific milestone after the scalar loop is stable. Approach C remains outside the initial project scope.

## 3. Scope

### In scope

- One whitelisted ODE template with immutable equation structure.
- A Pydantic configuration schema for domain, sampling, split case counts, candidate models, random seed, and acceptance criteria.
- Numerical labels generated with `scipy.integrate.solve_ivp`.
- An independent analytical solution used to verify numerical labels.
- Case-level train, validation, and test splitting by distinct `gamma` values.
- Three candidate estimators: polynomial response surface, Gaussian process regression, and `MLPRegressor`.
- Validation-based model selection and one final test evaluation.
- Saved preprocessing pipeline, selected model, metrics, manifest, model card, and diagnostic plots.
- Reloadable inference in a new Python process.
- Rejection of non-finite inputs and values outside the configured training domain.
- A deterministic CLI supporting validation, smoke execution, full execution, reporting, and prediction.
- Pytest coverage for scientific correctness, workflow invariants, persistence, and an end-to-end smoke run.

### Out of scope

- Parsing arbitrary equations.
- `eval`, `exec`, generated Python execution, or untrusted pickle loading.
- FEniCSx, CFD/FEM integration, DeepONet, FNO, PINNs, or PyTorch.
- A web interface, external LLM API, runtime multi-agent system, or Agents SDK.
- JSON Patch and multi-turn canonical state management.
- Optuna, MLflow, distributed execution, GPU training, and production deployment.
- Automatic extrapolation beyond the configured parameter domain.

## 4. Architecture

The project uses a `src` package layout and keeps scientific modules independent from orchestration.

```text
Natural-language request handled by Codex
                |
                v
         validated spec.json
                |
                v
      deterministic Python CLI
                |
     +----------+-----------+
     |          |           |
     v          v           v
 data source  trainers   acceptance gate
     |          |           |
     +----------+-----------+
                |
                v
        immutable run directory
                |
                v
      reloadable bounded inference
```

The CLI is orchestration only. Configuration validation, analytical reference calculation, numerical solving, data splitting, model construction, evaluation, artifact writing, and inference-domain checks each have a focused module and a typed interface.

## 5. Proposed Repository Structure

```text
surrogate-loop/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── scalar_ode.default.json
│   └── scalar_ode.smoke.json
├── docs/
│   └── superpowers/
│       ├── specs/
│       └── plans/
├── src/
│   └── surrogate_loop/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── domain.py
│       ├── data.py
│       ├── split.py
│       ├── models.py
│       ├── evaluation.py
│       ├── artifacts.py
│       ├── pipeline.py
│       └── inference.py
├── tests/
│   ├── test_config.py
│   ├── test_domain.py
│   ├── test_data.py
│   ├── test_split.py
│   ├── test_models.py
│   ├── test_artifacts.py
│   ├── test_inference.py
│   └── test_cli_smoke.py
└── runs/
```

`runs/` is generated output and is excluded from Git except for an optional placeholder file.

## 6. Component Responsibilities

### Configuration

`config.py` defines Pydantic models and rejects unknown fields. The problem identifier is the whitelist value `forced_reaction_scalar_endpoint_v1`; the equation is not accepted as executable input. Validation requires ordered finite bounds, positive sample counts, supported model names, and explicit acceptance thresholds.

The canonical full configuration uses seed `20260716`, 120 training cases, 40 validation cases, and 40 test cases. The smoke configuration uses the same seed with 24 training cases, 8 validation cases, and 8 test cases. Both sample `gamma` uniformly from `[-1, 1]` with NumPy's seeded generator and reject duplicate floating-point values before solving.

### Scientific domain

`domain.py` provides the analytical solution and the ODE right-hand side. At `gamma = 0`, it uses the continuous limit `u(t) = t^2 / 4`. For `abs(gamma * t) < 1e-4`, it uses a series expansion to avoid catastrophic cancellation; otherwise it uses `expm1` in the closed form.

`data.py` samples distinct `gamma` cases with a seeded generator, solves each case using `solve_ivp(method="DOP853", rtol=1e-10, atol=1e-12)`, and extracts `u(1)`. Every numerical label must agree with the analytical result to absolute tolerance `1e-9` and relative tolerance `1e-9`. Any failed solve, non-finite value, or tolerance violation stops the run before model training.

### Dataset splitting

`split.py` partitions distinct `gamma` cases into train, validation, and test sets before any preprocessing. No `gamma` value may occur in more than one split. Preprocessing is fitted only on the training split.

### Models and selection

`models.py` exposes a small registry returning scikit-learn-compatible pipelines:

- polynomial degrees 1, 2, and 3, each followed by `Ridge` with `alpha=1e-8`;
- `GaussianProcessRegressor` with `ConstantKernel * RBF`, `alpha=1e-10`, `normalize_y=True`, two optimizer restarts, and the configured seed;
- `MLPRegressor` with hidden layers `(32, 32)`, `tanh` activation, `alpha=1e-4`, `max_iter=2000`, early stopping on a 20% subset of the training partition, and the configured seed.

Each candidate is trained on the training split and ranked on the validation split using normalized root mean squared error. NRMSE is RMSE divided by `max(y_validation) - min(y_validation)`; a zero target range is a validation error. Mean absolute error and maximum absolute error are also reported. Ties within `1e-12` NRMSE are resolved by the fixed simplicity order polynomial degree 1, degree 2, degree 3, GPR, then MLP. The selected model is evaluated exactly once on the held-out test split.

### Acceptance and artifacts

`evaluation.py` compares final test metrics against explicit configuration thresholds. The canonical acceptance gate requires test NRMSE no greater than `0.03` and test maximum absolute error no greater than `0.01`. A technically completed run may have status `accepted` or `rejected`; poor accuracy is a valid rejected result, not a pipeline crash.

`artifacts.py` creates one run directory and writes files atomically. Required artifacts are the original request record, validated configuration, sampled-case metadata, split membership, per-model validation metrics, final test metrics, serialized trusted pipeline, manifest with software versions and hashes, model card, and plots.

### Inference

`inference.py` loads only artifacts produced and registered by this project, verifies the manifest, checks that `gamma` is finite and within the training range, and returns `u(1)` with model and run identifiers. Predictions outside the training range are rejected with a clear error.

### CLI

The package entry point provides:

```text
surrogate-loop validate --config <path>
surrogate-loop run --config <path> --smoke
surrogate-loop run --config <path>
surrogate-loop report --run-dir <path>
surrogate-loop predict --run-dir <path> --gamma <value>
```

`--smoke` requires the smoke configuration and verifies its bounded sample counts; it never silently mutates the full configuration.

## 7. Data and Control Flow

1. Codex records the user's request and creates or updates a JSON configuration using the fixed schema.
2. `validate` parses the configuration with unknown-field rejection.
3. `run` creates a pending run directory and records the validated configuration and environment metadata.
4. The seeded sampler creates distinct `gamma` cases.
5. The numerical generator computes `u(1)` and verifies every label against the analytical solution.
6. The cases are split into train, validation, and test partitions.
7. Each registered candidate is fitted on training data and scored on validation data.
8. The best validation candidate is frozen and evaluated once on test data.
9. The acceptance gate assigns `accepted` or `rejected` and completes the run artifacts.
10. `predict` verifies the run and input domain, reloads the selected pipeline, and returns a bounded prediction.

## 8. Error Handling

- Invalid configuration: exit non-zero before creating training artifacts and print field-level validation errors.
- Numerical solver or analytical-check failure: mark the run `failed`, preserve diagnostics, and do not train models.
- Candidate training failure: record the candidate error; continue only if at least one candidate remains valid.
- All candidates fail: mark the run `failed` and do not create a selected-model artifact.
- Accuracy threshold not met: mark the run `rejected`, preserve the best model for analysis, and make the failed threshold explicit.
- Corrupt or mismatched artifacts: refuse loading or prediction.
- Out-of-domain or non-finite prediction input: reject without invoking the estimator.
- Existing run identifier: never overwrite; create a new unique identifier.

## 9. Testing Strategy

Development follows test-driven implementation.

Required tests include:

- Configuration accepts the canonical files and rejects unknown fields, inverted bounds, invalid ratios, and unsupported models.
- The analytical solution satisfies the `gamma = 0` limit and agrees with high-accuracy numerical integration at negative, zero, near-zero, and positive `gamma` values.
- Sample generation is repeatable for a fixed seed.
- Split membership is disjoint and preprocessing never fits on validation or test values.
- Every model registry entry supports fit, predict, serialization, and reload.
- Selection uses validation metrics and final test metrics are produced only for the selected candidate.
- Saved and reloaded predictions agree within floating-point tolerance.
- Domain checks reject `gamma < low`, `gamma > high`, NaN, and infinity.
- A subprocess-level smoke test executes the CLI, produces all required artifacts, reloads the model in a new process, and performs one valid prediction.

## 10. Environment

The repository targets Windows and Python 3.11. Dependency and virtual-environment management use `uv` with a repository-local `.venv` and committed `uv.lock`.

Runtime dependencies:

- NumPy
- SciPy
- scikit-learn
- Pydantic 2
- Matplotlib
- joblib

Development dependencies:

- pytest
- pytest-cov
- Ruff

The initial workflow is CPU-only. The available RTX 4060 is intentionally unused because the selected models do not require GPU acceleration. PyTorch and CUDA are introduced only if a later approved milestone needs them.

## 11. Acceptance Criteria

The first scalar closed loop is complete when all of the following are true:

1. A fresh clone can create the locked environment and run the documented verification commands.
2. Ruff and the complete pytest suite pass.
3. The smoke CLI run completes and writes every required artifact.
4. Numerical labels agree with the analytical solution within the configured tolerance.
5. Train, validation, and test `gamma` memberships are disjoint.
6. The chosen model is selected by validation performance and evaluated once on the held-out test set.
7. A new Python process can reload the selected artifact and reproduce a saved prediction.
8. In-domain prediction succeeds, while out-of-domain and non-finite inputs are rejected.
9. The run manifest contains configuration, seed, dependency versions, data identity, selected model, metrics, and final status.
10. Codex can take a natural-language training request, produce a valid configuration, run the fixed CLI, and explain whether the acceptance gate passed.

## 12. Delivery Sequence

Implementation will be planned as reviewable tasks:

1. Repository and locked environment skeleton.
2. Configuration contract and canonical configurations.
3. Analytical and numerical data generation.
4. Leakage-free splitting and model registry.
5. Evaluation, selection, and acceptance gate.
6. Artifact persistence and bounded inference.
7. CLI integration and end-to-end smoke verification.
8. Codex operating instructions and final reproducibility documentation.

The next milestone after this design is approved is the full-trajectory contract `gamma -> u(t)`; it is not part of this implementation plan.
