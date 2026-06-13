# MLOps Workshop: Local ML to Production-Ready Pipelines

> **Duration:** 3 Hours | **Format:** Hands-on Demo + Practice | **Level:** Intermediate–Advanced

---

## Workshop Overview

This workshop walks you through building a production-grade MLOps pipeline from scratch. Starting with a raw local ML workflow, you'll progressively layer in experiment tracking, data versioning, model validation, and monitoring — ending with a fully reproducible, cloud-agnostic pipeline.

**By the end of this workshop, you will be able to:**
- Identify the gaps between local ML notebooks and production-ready systems
- Instrument a training pipeline with experiment tracking (MLflow or W&B)
- Version datasets and models using DVC
- Detect data drift and monitor model performance with Evidently
- Execute a complete, reproducible end-to-end MLOps pipeline

---

## Prerequisites Checklist

Before starting, confirm the following are ready on your machine:

- [ ] Python 3.8+ installed
- [ ] Jupyter Notebook or VS Code
- [ ] `pip install mlflow dvc evidently scikit-learn pandas numpy matplotlib`
- [ ] Git initialized in your project folder (`git init`)
- [ ] Dataset downloaded and placed in `/data/raw/` (CSV format recommended — e.g., UCI Wine Quality, Titanic, or Kaggle housing prices)
- [ ] Basic familiarity with training an ML model in Python

---

## Timetable at a Glance

| Section | Topic | Duration |
|---|---|---|
| 1 | Local ML Workflow Walkthrough | 20 min |
| 2 | Experiment Tracking with MLflow / W&B | 25 min |
| 3 | Data & Model Versioning with DVC | 25 min |
| Break | — | 20 min |
| 4 | Validation & Monitoring with Evidently | 40 min |
| 5 | End-to-End Pipeline Execution | 20 min |
| 6 | Conclusion & Q&A | 30 min |
| **Total** | | **180 min** |

---

## Section 1 — Local ML Workflow Walkthrough

**Duration: 20 minutes**

### Goal
Run a basic ML workflow and identify what's missing for production.

### Step 1.1 — Project Structure

Set up a clean project directory:

```
mlops-workshop/
├── data/
│   ├── raw/         ← your CSV dataset goes here
│   └── processed/
├── models/
├── reports/
├── src/
│   └── train.py
├── requirements.txt
└── README.md
```

### Step 1.2 — Baseline Training Script

Create `src/train.py`:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle

# 1. Load data
df = pd.read_csv("data/raw/dataset.csv")

# 2. Basic preprocessing
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"F1 Score: {f1_score(y_test, preds, average='weighted'):.4f}")

# 5. Save model (no versioning!)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to models/model.pkl")
```

Run it:
```bash
python src/train.py
```

### Discussion: Discussion: What's Missing?

After running the script, reflect on these questions:

| Gap | Problem |
|---|---|
| No experiment log | You can't recall what params produced what accuracy |
| No data versioning | Changing the CSV silently breaks reproducibility |
| No model registry | `model.pkl` is overwritten every run |
| No validation | No check that the model meets quality thresholds |
| No monitoring | No way to detect data drift in production |

> **Key Insight:** This script works locally today but is unauditable, unreproducible, and undeployable at scale.

---

## Section 2 — Experiment Tracking with MLflow

**Duration: 25 minutes**

### Goal
Instrument your training script to log parameters, metrics, and model artifacts — and compare runs visually.

### Step 2.1 — Install & Start MLflow

```bash
pip install mlflow
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "*" \
  --cors-allowed-origins "http://172.22.109.115:5000"
```

Open your browser at **http://172.22.109.115:5000**

### Step 2.2 — Instrument `train.py` with MLflow

> NOTE: **Fix applied:** The original script used hardcoded values — environment variables passed via `MAX_DEPTH=10 python src/train.py` were silently ignored. All config values now read from `os.environ` with fallback defaults. The deprecated `artifact_path` positional argument is also replaced with a keyword argument.

Replace your training script with this tracked version:

```python
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# --- Config: reads from env vars, falls back to defaults ---
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", 100))
MAX_DEPTH     = int(os.environ.get("MAX_DEPTH", 5))
TEST_SIZE     = float(os.environ.get("TEST_SIZE", 0.2))
RANDOM_STATE  = int(os.environ.get("RANDOM_STATE", 42))
DATA_PATH     = os.environ.get("DATA_PATH", "data/raw/dataset.csv")

# --- Load Data ---
df = pd.read_csv(DATA_PATH)
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# --- MLflow Experiment ---
mlflow.set_experiment("mlops-workshop")

run_name = f"rf-d{MAX_DEPTH}-n{N_ESTIMATORS}"

with mlflow.start_run(run_name=run_name):

    # Log parameters
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("data_path", DATA_PATH)

    # Train
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="weighted")

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log model as artifact (use keyword argument to avoid deprecation warning)
    mlflow.sklearn.log_model(model, artifact_path="random-forest-model")

    print(f"n_estimators={N_ESTIMATORS}  max_depth={MAX_DEPTH}")
    print(f"Run complete — Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
```

### Step 2.3 — Run Multiple Experiments

Vary hyperparameters across runs to generate comparison data:

```bash
# Run 1: baseline
python src/train.py

# Run 2: deeper trees
MAX_DEPTH=10 python src/train.py

# Run 3: more estimators
N_ESTIMATORS=200 python src/train.py

# Run 4: both changed
MAX_DEPTH=10 N_ESTIMATORS=200 python src/train.py
```

Each run will now produce **different accuracy values** and appear in the MLflow UI with distinct names (`rf-d5-n100`, `rf-d10-n100`, etc.).

### Step 2.4 — Compare in the MLflow UI

Go to **http://172.22.109.115:5000**, open your experiment, and:
1. Select all runs > click **Compare**
2. View parameter vs. metric scatter plots
3. Check the **Artifacts** tab for saved models

### [x] Checkpoint

- [ ] At least 3 runs visible in MLflow UI with different metrics
- [ ] Parameters, metrics, and model artifact logged per run
- [ ] Able to identify best-performing run by F1 score

---

## Section 3 — Data & Model Versioning with DVC

**Duration: 25 minutes**

### Goal
Version your dataset and model artifacts so any past experiment can be perfectly reproduced.

### Step 3.1 — Initialize DVC

```bash
pip install dvc
git init          # if not already done
dvc init
git add .dvc
git commit -m "Initialize DVC"
```

### Step 3.2 — Track Your Dataset

```bash
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "Track raw dataset v1 with DVC"
git tag -a "data-v1" -m "Initial dataset version"
```

DVC creates a `.dvc` pointer file — the actual data stays out of Git.

### Step 3.3 — Configure Remote Storage (Local Simulation)

```bash
# Simulate a remote using a local folder
mkdir /tmp/dvc-remote
dvc remote add -d localremote /tmp/dvc-remote
dvc push
```

In production, replace with:
```bash
dvc remote add -d s3remote s3://your-bucket/dvc-store
```

### Step 3.4 — Create a DVC Pipeline

Create `dvc.yaml`:

```yaml
stages:
  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/raw/dataset.csv
    params:
      - params.yaml:
          - n_estimators
          - max_depth
          - test_size
    outs:
      - models/model.pkl
    metrics:
      - reports/metrics.json:
          cache: false
```

Create `params.yaml`:

```yaml
n_estimators: 100
max_depth: 5
test_size: 0.2
random_state: 42
```

Run the pipeline:
```bash
dvc repro
```

### Step 3.5 — Simulate Version Switch

Make a change to your dataset (e.g., add a row or change a column), then:

```bash
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc
git commit -m "Dataset updated — v2"
git tag -a "data-v2" -m "Updated dataset"
dvc push
```

Restore v1:
```bash
git checkout data-v1
dvc checkout
python src/train.py  # Reproduces original results exactly
```

### Step 3.6 — Fix: output 'models/model.pkl' does not exist

If `dvc repro` fails with this error:

```
MLflow Run ID: 1b98af26c0314c909a9165b4323048a5
ERROR: failed to reproduce 'train': output 'models/model.pkl' does not exist
```

This means `train.py` runs successfully (MLflow logs the run) but never writes `models/model.pkl` or `reports/metrics.json` to disk — so DVC cannot find its expected outputs.

**Fix — update `src/train.py` to write both files:**

Add `os.makedirs` and both save blocks at the end of the script, inside the `mlflow.start_run` block:

```python
import os
import json
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", 100))
MAX_DEPTH     = int(os.environ.get("MAX_DEPTH", 5))
TEST_SIZE     = float(os.environ.get("TEST_SIZE", 0.2))
RANDOM_STATE  = int(os.environ.get("RANDOM_STATE", 42))
DATA_PATH     = os.environ.get("DATA_PATH", "data/raw/dataset.csv")

df = pd.read_csv(DATA_PATH)
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

mlflow.set_experiment("mlops-workshop")
run_name = f"rf-d{MAX_DEPTH}-n{N_ESTIMATORS}"

with mlflow.start_run(run_name=run_name):
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("data_path", DATA_PATH)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, artifact_path="random-forest-model")

    # Save model for DVC
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save metrics for DVC
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as mf:
        json.dump({"accuracy": round(acc, 4), "f1_score": round(f1, 4)}, mf, indent=2)

    print(f"n_estimators={N_ESTIMATORS}  max_depth={MAX_DEPTH}")
    print(f"Run complete — Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
    print("Model saved   > models/model.pkl")
    print("Metrics saved > reports/metrics.json")
```

**Verify the files are created before running DVC:**

```bash
python src/train.py
ls models/model.pkl        # must exist
ls reports/metrics.json    # must exist
```

**Then run DVC:**

```bash
dvc repro
dvc push
git add .
git commit -m "Fix train.py to write model and metrics for DVC"
```

**Root cause:** `dvc.yaml` declares `models/model.pkl` and `reports/metrics.json` as outputs. If `train.py` does not write them, DVC considers the stage failed even if MLflow logged the run successfully. MLflow and DVC track the same training run independently — MLflow tracks metadata, DVC tracks files on disk.

### [x] Checkpoint

- [ ] Dataset tracked with `dvc add` and committed to Git
- [ ] Pipeline runs via `dvc repro`
- [ ] `models/model.pkl` and `reports/metrics.json` both exist after run
- [ ] Successfully switched between dataset versions

### Common Issue — DVC Cache Errors

If you see any of the following errors after `dvc checkout` or `dvc repro`:

```
WARNING: No file hash info found for 'models/model.pkl'
ERROR: Checkout failed for following targets: models/model.pkl
FileNotFoundError: No such file or directory: 'data/raw/dataset.csv'
```

This means DVC is tracking files that were never pushed to its cache. Fix in order:

**Step 1 — Restore the dataset**

If `data/raw/dataset.csv` was deleted by `dvc checkout`, regenerate it:

```python
import pandas as pd, numpy as np
np.random.seed(42); n = 1000
df = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'income': np.random.normal(50000, 15000, n).clip(15000, 120000).round(2),
    'credit_score': np.random.randint(300, 850, n),
    'loan_amount': np.random.normal(15000, 8000, n).clip(1000, 50000).round(2),
    'employment_years': np.random.randint(0, 30, n),
    'num_accounts': np.random.randint(1, 10, n),
    'debt_ratio': np.random.uniform(0.1, 0.9, n).round(4),
})
prob = 1/(1+np.exp(-(-0.003*df.credit_score + 0.00001*df.loan_amount
         - 0.000005*df.income + 1.5*df.debt_ratio
         - 0.05*df.employment_years + 2.0)))
df['target'] = (np.random.rand(n) < prob).astype(int)
df.to_csv('data/raw/dataset.csv', index=False)
print('Done', df.shape)
```

**Step 2 — Re-add to DVC and push to cache**

```bash
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc
git commit -m "Re-track dataset with valid cache"
dvc push
```

**Step 3 — Run the pipeline to regenerate the model**

```bash
dvc repro
dvc push
```

**Step 4 — Verify**

```bash
dvc status      # should say: Data and pipelines are up to date
dvc checkout    # should complete with no errors
```

**Root cause summary:**

| Error | Cause | Fix |
|---|---|---|
| `dataset.csv` deleted | `dvc checkout` replaced it with cached version but cache was empty | Re-add and push the file |
| `model.pkl` missing hash | Model was never run through `dvc repro`, only saved manually | Run `dvc repro` to let DVC manage it |
| Cache out of date | `dvc push` was never run after `dvc add` | Always `dvc push` after `dvc add` |

**Correct workflow going forward — always follow this order:**

```bash
# After any change to data or code:
dvc repro       # regenerates outputs
dvc push        # saves to cache
git add .
git commit -m "your message"
```

---

## Break — 20 Minutes

---

## Section 4 — Validation & Monitoring with Evidently

**Duration: 40 minutes**

### Goal
Run model quality validation, detect data drift, and generate monitoring reports.

### Step 4.1 — Install Evidently

```bash
pip install evidently
```

Check your version after install — the imports differ by version:

```bash
pip show evidently | grep Version
```

| Version | Import path |
|---|---|
| Below 0.4 | `pip install --upgrade evidently` to fix |
| 0.4 - 0.6 | `from evidently.report import Report` |
| 0.7+ | `from evidently.legacy.report import Report` (used in this workshop) |

### Step 4.2 — Simulate Reference vs. Production Data

Create `src/generate_drift.py` to simulate data drift:

```python
import pandas as pd
import numpy as np

# Load original (reference) data
df = pd.read_csv("data/raw/dataset.csv")

# Create "production" data with artificial drift
df_prod = df.copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "target"]

# Shift means and add noise to simulate drift
for col in numeric_cols[:3]:   # drift first 3 numeric features
    df_prod[col] = df_prod[col] * 1.3 + np.random.normal(0, 0.5, len(df_prod))

df.to_csv("data/processed/reference.csv", index=False)
df_prod.to_csv("data/processed/production.csv", index=False)
print("Reference and production datasets saved.")
```

Run it:
```bash
python src/generate_drift.py
```

### Step 4.3 — Data Drift Report

Create `src/monitoring.py`.

NOTE: Evidently 0.7+ moved the classic API under `evidently.legacy`. All imports below use this path. If you are on 0.4-0.6, replace `evidently.legacy.report` with `evidently.report` and `evidently.legacy.metric_preset` with `evidently.metric_preset`.

```python
import os
import pandas as pd
import pickle
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping

os.makedirs("reports", exist_ok=True)

# Load datasets
reference  = pd.read_csv("data/processed/reference.csv")
production = pd.read_csv("data/processed/production.csv")

ref_features  = reference.drop("target", axis=1)
prod_features = production.drop("target", axis=1)

# --- Data Drift Report ---
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=ref_features, current_data=prod_features)
drift_report.save_html("reports/data_drift_report.html")
print("Drift report saved   > reports/data_drift_report.html")

# --- Data Quality Report ---
quality_report = Report(metrics=[DataQualityPreset()])
quality_report.run(reference_data=ref_features, current_data=prod_features)
quality_report.save_html("reports/data_quality_report.html")
print("Quality report saved > reports/data_quality_report.html")
```

```bash
python src/monitoring.py
```

Open `reports/data_drift_report.html` in your browser.

### Step 4.4 — Model Performance Monitoring

Add model prediction tracking to the same `src/monitoring.py` file, appended after the quality report block:

```python
# --- Model Performance Report ---
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

reference["prediction"]  = model.predict(ref_features)
production["prediction"] = model.predict(prod_features)

column_mapping = ColumnMapping(target="target", prediction="prediction")

perf_report = Report(metrics=[ClassificationPreset()])
perf_report.run(
    reference_data=reference,
    current_data=production,
    column_mapping=column_mapping
)
perf_report.save_html("reports/model_performance_report.html")
print("Performance report   > reports/model_performance_report.html")
```

The complete `src/monitoring.py` with all three reports combined:

```python
import os
import pandas as pd
import pickle
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping

os.makedirs("reports", exist_ok=True)

reference  = pd.read_csv("data/processed/reference.csv")
production = pd.read_csv("data/processed/production.csv")

ref_features  = reference.drop("target", axis=1)
prod_features = production.drop("target", axis=1)

drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=ref_features, current_data=prod_features)
drift_report.save_html("reports/data_drift_report.html")
print("Drift report saved   > reports/data_drift_report.html")

quality_report = Report(metrics=[DataQualityPreset()])
quality_report.run(reference_data=ref_features, current_data=prod_features)
quality_report.save_html("reports/data_quality_report.html")
print("Quality report saved > reports/data_quality_report.html")

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

reference["prediction"]  = model.predict(ref_features)
production["prediction"] = model.predict(prod_features)

column_mapping = ColumnMapping(target="target", prediction="prediction")

perf_report = Report(metrics=[ClassificationPreset()])
perf_report.run(
    reference_data=reference,
    current_data=production,
    column_mapping=column_mapping
)
perf_report.save_html("reports/model_performance_report.html")
print("Performance report   > reports/model_performance_report.html")
```

### Step 4.5 — Validation Gates (Automated Quality Check)

Create `src/validate.py` to enforce quality thresholds:

```python
import json
import sys

THRESHOLDS = {
    "accuracy": 0.80,
    "f1_score": 0.78,
}

with open("reports/metrics.json") as f:
    metrics = json.load(f)

failed = []
for metric, threshold in THRESHOLDS.items():
    value = metrics.get(metric, 0)
    status = "[x] PASS" if value >= threshold else "❌ FAIL"
    print(f"{status} | {metric}: {value:.4f} (threshold: {threshold})")
    if value < threshold:
        failed.append(metric)

if failed:
    print(f"\nValidation FAILED on: {failed}")
    sys.exit(1)
else:
    print("\nAll validation checks passed. Model is deployment-ready.")
```

```bash
python src/validate.py
```

### [x] Checkpoint

- [ ] Data drift report generated in `reports/`
- [ ] Feature distribution shift visible for drifted columns
- [ ] Model performance report comparing reference vs. production
- [ ] Validation gate script enforces thresholds and exits non-zero on failure

---

## Section 5 — End-to-End Pipeline Execution

**Duration: 20 minutes**

### Goal
Execute the full pipeline from versioned data through to monitoring report — in a single reproducible sequence.

### Step 5.1 — Full Pipeline Script

Create `run_pipeline.sh`:

```bash
#!/bin/bash
set -e   # Exit immediately on any error

echo "=============================="
echo " MLOps Pipeline Starting"
echo "=============================="

# Step 1: Ensure data is checked out
echo "[1/5] Checking out data version..."
dvc checkout

# Step 2: Run DVC pipeline (train + preprocess)
echo "[2/5] Running DVC pipeline..."
dvc repro

# Step 3: Run validation gates
echo "[3/5] Validating model quality..."
python src/validate.py

# Step 4: Run monitoring reports
echo "[4/5] Generating monitoring reports..."
python src/monitoring.py

# Step 5: Push artifacts
echo "[5/5] Pushing artifacts to DVC remote..."
dvc push

echo ""
echo "=============================="
echo " Pipeline Complete [x]"
echo "=============================="
echo "Artifacts:"
echo "  Model      > models/model.pkl"
echo "  Metrics    > reports/metrics.json"
echo "  Drift      > reports/data_drift_report.html"
echo "  Quality    > reports/data_quality_report.html"
echo "  MLflow UI  > http://172.22.109.115:5000"
```

Make it executable and run:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Step 5.2 — Verify Reproducibility

Clean the model output and re-run:

```bash
rm -f models/model.pkl
./run_pipeline.sh
```

The pipeline should produce the **exact same metrics** as the previous run.

### Step 5.3 — What Makes This Cloud-Ready?

| Principle | How It's Implemented |
|---|---|
| No hard-coded paths | Paths driven via `params.yaml` and env vars |
| Stateless execution | Pipeline reruns cleanly with no side effects |
| Storage independence | DVC remote can point to S3, GCS, or Azure Blob |
| Portable metadata | MLflow tracking URI is configurable |
| Separation of concerns | Code (Git) / Data (DVC) / Models (DVC) / Metadata (MLflow) |

### Step 5.4 — Migration to Cloud (Conceptual Map)

```
Local                          Cloud Equivalent
─────────────────────────────────────────────────────
/tmp/dvc-remote          >     S3 / GCS / Azure Blob
mlflow server (local)    >     MLflow on EC2 / Databricks
python run_pipeline.sh   >     Airflow / Kubeflow / SageMaker Pipelines
models/model.pkl         >     MLflow Model Registry
```

> The pipeline logic doesn't change — only the infrastructure wiring does.

### [x] Checkpoint

- [ ] Full pipeline runs end-to-end with `run_pipeline.sh`
- [ ] Reproducibility verified by deleting and re-running
- [ ] All 4 artifacts present: model, metrics, drift report, quality report
- [ ] Cloud migration path is clear conceptually

---

## Section 6 — Conclusion & Q&A

**Duration: 30 minutes**

### What Changed: Local ML > MLOps

| Capability | Local Notebook | This Pipeline |
|---|---|---|
| Parameter tracking | [no] None | [x] MLflow |
| Experiment history | [no] None | [x] MLflow UI |
| Dataset versioning | [no] None | [x] DVC |
| Model versioning | [no] Overwritten | [x] DVC + MLflow |
| Reproducibility | [no] Manual/fragile | [x] `dvc repro` |
| Quality gates | [no] None | [x] `validate.py` |
| Drift detection | [no] None | [x] Evidently |
| Cloud portability | [no] Local only | [x] Platform-agnostic |

### Do's and Don'ts

**Do:**
- Always version data alongside code changes
- Log every meaningful hyperparameter and metric
- Use validation gates before promoting a model
- Keep storage, compute, and orchestration concerns separate

**Don't:**
- Hard-code file paths or credentials in scripts
- Overwrite model files without versioning
- Skip monitoring in production — drift happens silently
- Treat reproducibility as optional

### Tools Summary

| Tool | Purpose |
|---|---|
| **MLflow** | Experiment tracking, model registry |
| **Weights & Biases** | Alternative to MLflow (cloud-native, richer UI) |
| **DVC** | Data and model versioning, pipeline orchestration |
| **Evidently** | Data drift detection, model monitoring reports |
| **Git** | Code versioning (works in tandem with DVC) |

### Further Reading

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Evidently Documentation](https://docs.evidentlyai.com)
- [Weights & Biases](https://docs.wandb.ai)
- [Made With ML — MLOps](https://madewithml.com)

---

## Appendix — Troubleshooting

**MLflow UI not loading:**
```bash
# Ensure port 5000 is free
lsof -i :5000
mlflow server --host 0.0.0.0 --port 5001 --allowed-hosts "*"
```

**DVC checkout fails:**
```bash
# Pull data from remote first
dvc pull
dvc checkout
```

**DVC cache missing — dataset deleted or model.pkl has no hash:**

Symptoms: `WARNING: No file hash info found`, `Checkout failed for models/model.pkl`, or `FileNotFoundError: data/raw/dataset.csv` immediately after `dvc checkout`.

```bash
# 1. Restore and re-track the dataset (see Section 3 — Common Issue for full script)
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc
git commit -m "Re-track dataset with valid cache"
dvc push

# 2. Regenerate the model via pipeline (never save model.pkl manually)
dvc repro
dvc push

# 3. Confirm everything is clean
dvc status
```

Rule: always run `dvc push` immediately after `dvc add` or `dvc repro`. Never copy `model.pkl` manually — let `dvc repro` create it.

**Evidently import error — ModuleNotFoundError or ImportError:**

Evidently 0.7+ moved the classic API under `evidently.legacy`. Check your version first:

```bash
pip show evidently | grep Version
```

If you see `ModuleNotFoundError: No module named 'evidently.report'` or `ImportError: cannot import name 'ColumnMapping' from 'evidently'`, update all imports in `src/monitoring.py`:

```python
# Old imports (Evidently 0.4 - 0.6) — causes errors on 0.7+
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

# Fixed imports (Evidently 0.7+)
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
```

If the package is missing entirely:
```bash
pip install --upgrade evidently
```

**Pipeline exits with validation failure:**
- Check `reports/metrics.json` for actual metric values
- Adjust thresholds in `src/validate.py` to match your dataset's baseline

**Model produces different results across runs:**
- Ensure `random_state` is set on all sklearn objects
- Confirm the same dataset version is checked out via `dvc checkout`

**All experiment runs show identical metrics:**
- Ensure `train.py` reads env vars with `os.environ.get()` — hardcoded values ignore shell exports
- Verify with: `N_ESTIMATORS=200 python -c "import os; print(os.environ.get('N_ESTIMATORS'))"`

---

*Workshop material for MLOps Demo session. Designed to be cloud-agnostic and reproducible across environments.*
