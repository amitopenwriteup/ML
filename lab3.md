A **Random Forest** is a machine learning algorithm that combines many **decision trees** and lets them vote on the final answer.

Think of it like asking 100 people for their opinion instead of relying on just one person.

---

## Step 1: What is a Decision Tree?

A decision tree makes predictions by asking a series of questions.

For example, to predict whether a customer will buy a product:

```text
Age > 30?
├── Yes
│   ├── Salary > 50,000?
│   │   ├── Yes → Buy
│   │   └── No  → Don't Buy
│
└── No
    └── Don't Buy
```

The tree keeps splitting the data based on conditions until it reaches a prediction.

---

## Step 2: Why One Tree Isn't Enough

A single decision tree can easily **overfit**.

Imagine learning from only a few examples:

```text
Tree says:
"If age is 31, always buy."
```

That rule might work for training data but fail on new data.

---

## Step 3: Random Forest Creates Many Trees

Instead of one tree:

```text
Tree 1
Tree 2
Tree 3
...
Tree 100
```

Each tree is trained on a slightly different random subset of the data.

That's where the **"Random"** in Random Forest comes from.

---

## Step 4: Voting

Suppose we want to predict whether a customer will buy:

```text
Tree 1 → Yes
Tree 2 → Yes
Tree 3 → No
Tree 4 → Yes
Tree 5 → Yes
```

Result:

```text
Yes wins 4 votes to 1
```

Final prediction:

```text
Buy = Yes
```

This is called **majority voting**.

---

## For Regression Problems

If predicting a number instead of a category:

Example:

```text
Tree 1 → 100
Tree 2 → 120
Tree 3 → 110
```

Average:

```text
(100 + 120 + 110) / 3 = 110
```

Final prediction:

```text
110
```

---

## Why It Works Well

A single tree may make mistakes.

```text
Tree 1 → Wrong
Tree 2 → Wrong
Tree 3 → Correct
...
```

When many trees vote, individual mistakes tend to cancel out.

This usually gives:

* Better accuracy
* Less overfitting
* More stable predictions

---

## Parameters in Your Code

You used:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
```

### n_estimators

```python
n_estimators=100
```

Means:

```text
Build 100 trees
```

More trees generally improve stability but take longer to train.

---

### max_depth

```python
max_depth=5
```

Limits how deep each tree can grow.

```text
Depth 1 → 1 question
Depth 5 → up to 5 levels of questions
```

Small depth:

* Simpler model
* Less overfitting

Large depth:

* More complex model
* Can overfit

---

### random_state

```python
random_state=42
```

Ensures you get the same random choices every time you run the code.

---

## Simple Example

Suppose we're predicting whether a student passes an exam.

Features:

```text
Hours Studied
Attendance
Assignments Submitted
```

Input:

```text
Hours Studied = 8
Attendance = 90%
Assignments = Yes
```

Trees vote:

```text
Tree 1 → Pass
Tree 2 → Pass
Tree 3 → Fail
Tree 4 → Pass
Tree 5 → Pass
```

Result:

```text
Pass (4 votes)
```

Think of your code as **teaching a student and then giving an exam**.

---

## Step 1: Read the data

```python
df = pd.read_csv("data/raw/dataset.csv")
```

This loads the CSV file into memory.

Example:

| age | income | credit_score | target |
| --- | ------ | ------------ | ------ |
| 25  | 30000  | 700          | 1      |
| 40  | 50000  | 600          | 0      |

The data becomes a table called `df`.

---

## Step 2: Separate Questions and Answers

```python
X = df.drop("target", axis=1)
y = df["target"]
```

Imagine a teacher giving:

### Questions (X)

```text
age
income
credit_score
loan_amount
...
```

### Answers (y)

```text
target
```

Example:

| Age | Income | Target |
| --- | ------ | ------ |
| 25  | 30000  | 1      |
| 40  | 50000  | 0      |

The model's job is:

```text
Given Age + Income + Credit Score
Predict Target
```

---

## Step 3: Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Suppose you have 1000 rows.

```text
800 rows → Training
200 rows → Testing
```

Training data teaches the model.

Testing data checks whether it learned correctly.

---

## Step 4: Create the Model

```python
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
```

### Random Forest = 100 decision trees

Imagine asking 100 experts:

```text
Expert 1 → Yes
Expert 2 → No
Expert 3 → Yes
...
```

The majority answer wins.

That's exactly what Random Forest does.

```python
n_estimators=100
```

means:

```text
Create 100 trees
```

```python
max_depth=5
```

means:

```text
Each tree can ask at most 5 levels of questions
```

---

## Step 5: Train the Model

```python
model.fit(X_train, y_train)
```

This is the learning step.

The model studies examples like:

```text
Age=25 Income=30000 → Target=1
Age=40 Income=50000 → Target=0
Age=35 Income=45000 → Target=1
```

and learns patterns.

---

## Step 6: Take the Exam

```python
preds = model.predict(X_test)
```

Now the model sees new rows it never saw before.

Example:

```text
Age=30 Income=40000
```

and predicts:

```text
Target = 1
```

---

## Step 7: Check Marks

### Accuracy

```python
accuracy_score(y_test, preds)
```

Measures:

```text
How many answers were correct?
```

Example:

```text
200 questions
160 correct
```

Accuracy:

```text
160 / 200 = 80%
```

---

### F1 Score

```python
f1_score(y_test, preds)
```

Another way to measure quality.

Range:

```text
0 = bad
1 = perfect
```

---

## Step 8: Save the Trained Model

```python
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
```

After learning, save the model to a file:

```text
models/model.pkl
```

Think of it as:

```text
Train once
Save the brain
Reuse later
```

Without saving:

```text
Every time you start
you must train again
```

With saving:

```text
Load the trained model
and predict immediately
```

---

## Complete Flow

```text
dataset.csv
      ↓
Load Data
      ↓
Separate Features & Target
      ↓
Split Train/Test
      ↓
Train Random Forest
      ↓
Predict on Test Data
      ↓
Calculate Accuracy & F1
      ↓
Save model.pkl
```

In one sentence:

**This script teaches a Random Forest model using historical data, tests how well it learned, and saves the trained model for future use.**


---

## Why Random Forest Is Popular

✅ Easy to use
✅ Works well on many datasets
✅ Handles missing patterns well
✅ Usually needs little tuning
✅ Good baseline model for classification and regression

Because of these advantages, Random Forest is often one of the first models data scientists try before moving to more advanced methods like XGBoost, LightGBM, or neural networks.
---------------------------

Sure. Think of **MLflow as a notebook for your machine learning experiments**.

Without MLflow:

```text
Run model
↓
Accuracy = 0.61

Change parameter
↓
Run again
↓
Accuracy = 0.65

Change parameter
↓
Run again
↓
Accuracy = 0.63
```

After a few runs, you forget:

* Which parameters produced 0.65?
* Which model file belongs to which run?
* Which run was best?

MLflow solves this.

---

# Step 2.1 — Start MLflow

Install:

```bash
pip install mlflow
```

Start MLflow server:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "*" \
  --cors-allowed-origins "http://172.22.109.115:5000"
```

Open:

```text
http://172.22.109.115:5000
```

Initially you'll see no experiments.

---

# Step 2.2 — Add MLflow to train.py

Previously your script only did:

```text
Train model
Print accuracy
Save model
```

Now MLflow additionally records:

```text
Parameters
Metrics
Artifacts
```

---

## Parameters

These are settings used during training.

Example:

```python
mlflow.log_param("n_estimators", N_ESTIMATORS)
mlflow.log_param("max_depth", MAX_DEPTH)
```

Suppose:

```bash
MAX_DEPTH=10
N_ESTIMATORS=200
```

MLflow stores:

| Parameter    | Value |
| ------------ | ----- |
| max_depth    | 10    |
| n_estimators | 200   |

---

## Metrics

These measure performance.

```python
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("f1_score", f1)
```

Example:

| Metric   | Value |
| -------- | ----- |
| accuracy | 0.67  |
| f1_score | 0.65  |

---

## Artifacts

Artifacts are files produced by the run.

```python
mlflow.sklearn.log_model(
    model,
    artifact_path="random-forest-model"
)
```

MLflow stores:

```text
trained model
```

instead of just saving:

```text
models/model.pkl
```

locally.

---

# Why Environment Variables?

Earlier you had:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5
)
```

Hardcoded.

So:

```bash
MAX_DEPTH=10 python train.py
```

did nothing.

Now:

```python
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", 5))
```

means:

```text
If MAX_DEPTH exists
    use it
Else
    use 5
```

---

Example:

```bash
MAX_DEPTH=10 python train.py
```

Model becomes:

```python
RandomForestClassifier(
    max_depth=10
)
```

---

# Step 2.3 — Run Experiments

### Run 1

```bash
python src/train.py
```

Logs:

```text
max_depth=5
n_estimators=100
```

Run name:

```text
rf-d5-n100
```

---

### Run 2

```bash
MAX_DEPTH=10 python src/train.py
```

Logs:

```text
max_depth=10
n_estimators=100
```

Run name:

```text
rf-d10-n100
```

---

### Run 3

```bash
N_ESTIMATORS=200 python src/train.py
```

Logs:

```text
max_depth=5
n_estimators=200
```

Run name:

```text
rf-d5-n200
```

---

### Run 4

```bash
MAX_DEPTH=10 N_ESTIMATORS=200 python src/train.py
```

Logs:

```text
max_depth=10
n_estimators=200
```

Run name:

```text
rf-d10-n200
```

---

# What MLflow Stores

For each run:

```text
Run Name
Parameters
Metrics
Model
Timestamp
Run ID
```

Example:

```text
Run: rf-d10-n200

Parameters:
max_depth=10
n_estimators=200

Metrics:
accuracy=0.67
f1_score=0.65

Artifacts:
random-forest-model
```

---

# Step 2.4 — Compare Runs

Open MLflow UI.

You may see:

| Run         | Depth | Trees | Accuracy | F1   |
| ----------- | ----- | ----- | -------- | ---- |
| rf-d5-n100  | 5     | 100   | 0.61     | 0.61 |
| rf-d10-n100 | 10    | 100   | 0.64     | 0.63 |
| rf-d5-n200  | 5     | 200   | 0.63     | 0.62 |
| rf-d10-n200 | 10    | 200   | 0.67     | 0.65 |

Now it's easy to answer:

### Which run is best?

Look at F1:

```text
rf-d10-n200
F1 = 0.65
```

Best run.

---

# What "Compare" Does

Select multiple runs.

Click:

```text
Compare
```

MLflow shows:

```text
Parameters vs Accuracy
Parameters vs F1
Charts
Tables
```

Instead of manually checking logs.

---

# Artifacts Tab

Click a run.

Open:

```text
Artifacts
```

You'll find:

```text
random-forest-model
```

This is the trained model for that specific run.

Each run gets its own saved model.

---

# Checkpoint Explained

By the end you should have:

✅ At least 3 runs visible

```text
rf-d5-n100
rf-d10-n100
rf-d5-n200
```

✅ Parameters logged

```text
max_depth
n_estimators
```

✅ Metrics logged

```text
accuracy
f1_score
```

✅ Model artifact saved

```text
random-forest-model
```

✅ Able to identify best run

```text
Highest F1 Score
```

So the key idea is:

**Random Forest trains the model, while MLflow keeps a complete history of every experiment so you can compare runs and find the best model.**


ection 3 — Data & Model Versioning with DVC
Duration: 25 minutes

Goal
Version your dataset and model artifacts so any past experiment can be perfectly reproduced.

Step 3.1 — Initialize DVC
pip install dvc
git init          # if not already done
dvc init
git add .dvc
git commit -m "Initialize DVC"
Step 3.2 — Track Your Dataset
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "Track raw dataset v1 with DVC"
git tag -a "data-v1" -m "Initial dataset version"
DVC creates a .dvc pointer file — the actual data stays out of Git.

Step 3.3 — Configure Remote Storage (Local Simulation)
# Simulate a remote using a local folder
mkdir /tmp/dvc-remote
dvc remote add -d localremote /tmp/dvc-remote
dvc push
In production, replace with:

dvc remote add -d s3remote s3://your-bucket/dvc-store
Step 3.4 — Create a DVC Pipeline
Create dvc.yaml:

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
Create params.yaml:

n_estimators: 100
max_depth: 5
test_size: 0.2
random_state: 42
Run the pipeline:

dvc repro
Step 3.5 — Simulate Version Switch
Make a change to your dataset (e.g., add a row or change a column), then:

dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc
git commit -m "Dataset updated — v2"
git tag -a "data-v2" -m "Updated dataset"
dvc push
Restore v1:

git checkout data-v1
dvc checkout
python src/train.py  # Reproduces original results exactly
Step 3.6 — Fix: output 'models/model.pkl' does not exist
If dvc repro fails with this error:

MLflow Run ID: 1b98af26c0314c909a9165b4323048a5
ERROR: failed to reproduce 'train': output 'models/model.pkl' does not exist
This means train.py runs successfully (MLflow logs the run) but never writes models/model.pkl or reports/metrics.json to disk — so DVC cannot find its expected outputs.

Fix — update src/train.py to write both files:

Add os.makedirs and both save blocks at the end of the script, inside the mlflow.start_run block:

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
Verify the files are created before running DVC:

python src/train.py
ls models/model.pkl        # must exist
ls reports/metrics.json    # must exist
Then run DVC:

dvc repro
dvc push
git add .
git commit -m "Fix train.py to write model and metrics for DVC"
Root cause: dvc.yaml declares models/model.pkl and reports/metrics.json as outputs. If train.py does not write them, DVC considers the stage failed even if MLflow logged the run successfully. MLflow and DVC track the same training run independently — MLflow tracks metadata, DVC tracks files on disk.

[x] Checkpoint
 Dataset tracked with dvc add and committed to Git
 Pipeline runs via dvc repro
 models/model.pkl and reports/metrics.json both exist after run
 Successfully switched between dataset versions
Common Issue — DVC Cache Errors
If you see any of the following errors after dvc checkout or dvc repro:

WARNING: No file hash info found for 'models/model.pkl'
ERROR: Checkout failed for following targets: models/model.pkl
FileNotFoundError: No such file or directory: 'data/raw/dataset.csv'


