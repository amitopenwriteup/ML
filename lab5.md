# Understanding Data & Model Versioning with DVC (For Beginners)

## Why does this section exist?

When you train a machine learning model, three things change over time:

1. Your **code** (`train.py`, hyperparameters, etc.)
2. Your **data** (the dataset you trained on)
3. Your **model output** (the trained model file, metrics)

Git is great at tracking #1 (code), but it's *terrible* at tracking #2 and #3 — datasets and model files are often large binary blobs, and Git was never designed for that.

**DVC (Data Version Control)** solves this. Think of it as "Git for data and models." It lets you say:

> "This exact code + this exact dataset version + these exact parameters produced this exact model."

That's the whole point of **reproducibility** — being able to go back in time and recreate any past experiment perfectly.

---

## The Big Picture: How DVC and Git Work Together

A useful mental model for newbies:

| Tool | What it tracks | Where the real files live |
|------|-----------------|----------------------------|
| **Git** | Code, small config files, and small `.dvc` pointer files | GitHub / your Git server |
| **DVC** | Large files (datasets, models, metrics) | A separate "remote" storage (local folder, S3, etc.) |

So instead of committing `dataset.csv` (which might be hundreds of MB) directly to Git, DVC creates a tiny **pointer file** called `dataset.csv.dvc`. Git tracks that small pointer file. The actual `dataset.csv` lives in DVC's storage ("the cache" or "the remote").

It's like Git keeps a "claim ticket" for your luggage, and DVC is the baggage counter that actually holds the bag.

---

## Step-by-Step Walkthrough

### Step 3.1 — `dvc init`

This sets up DVC inside your existing Git repo (similar to how `git init` sets up Git). It creates a `.dvc` folder for DVC's internal config. You commit this setup to Git so your teammates get the same configuration.

### Step 3.2 — `dvc add data/raw/dataset.csv`

This is the key moment. DVC:

- Copies your dataset into its internal cache
- Creates a small pointer file: `data/raw/dataset.csv.dvc`
- Adds `dataset.csv` to `.gitignore` so Git *won't* try to track the big file itself

You then commit the **pointer file** (not the dataset) to Git, and tag it `data-v1` so you can refer back to "version 1 of the data" later.

**Analogy:** Imagine writing a sticky note that says "Dataset v1 is stored in locker #42" and pinning that note to your project. The note (tiny) goes in Git. The dataset (heavy) goes in the locker (DVC remote).

### Step 3.3 — Configure Remote Storage

DVC needs somewhere to actually store the data files. In this workshop, that's a local folder (`/tmp/dvc-remote`) to keep things simple. In a real company, this would be cloud storage like Amazon S3, Google Cloud Storage, or Azure Blob Storage.

`dvc push` uploads your tracked files (the actual data) to this remote storage — same idea as `git push`, but for the heavy files.

### Step 3.4 — Create a DVC Pipeline (`dvc.yaml` + `params.yaml`)

This is where DVC becomes more than just "Git for big files" — it becomes a **pipeline manager**.

`dvc.yaml` describes one or more **stages**. Each stage says:

- `cmd`: what command to run (e.g., `python src/train.py`)
- `deps`: what inputs this stage depends on (code + data)
- `params`: which hyperparameters affect this stage (read from `params.yaml`)
- `outs`: what files this stage is expected to *produce*
- `metrics`: where to find performance numbers (accuracy, F1, etc.)

`params.yaml` is just a clean, central place to store hyperparameters like `n_estimators`, `max_depth`, and `test_size`, instead of hardcoding them inside `train.py`.

**Why this matters:** Once this is set up, DVC knows the full chain: *these inputs + these parameters → these outputs*. If any input or parameter changes, DVC knows the stage needs to be re-run.

### Step 3.5 — Make `train.py` Actually Produce the Declared Outputs

This is the most important lesson in the whole section, and it's a very common beginner mistake.

`dvc.yaml` *promises* that running the `train` stage will produce:

- `models/model.pkl`
- `reports/metrics.json`

But a promise isn't enough — the code has to **actually create those files**, in those exact paths, every time it runs. If `train.py` only logs results to MLflow but never writes `model.pkl` or `metrics.json` to disk, DVC will say:

```
ERROR: failed to reproduce 'train': output 'models/model.pkl' does not exist
```

The fix is straightforward:

1. Inside the `with mlflow.start_run(...)` block, after training the model:
   - `os.makedirs("models", exist_ok=True)` then save the model with `pickle.dump(...)`
   - `os.makedirs("reports", exist_ok=True)` then save metrics as JSON

**The key takeaway — MLflow and DVC are two separate filing systems that happen to watch the same event:**

- **MLflow** answers: *"What experiments did I run, and what were the results?"* (a logbook/dashboard)
- **DVC** answers: *"What files on disk were produced by which version of code/data/params?"* (a file-tracking system)

They don't automatically share information. If your script only "tells" MLflow about the model but doesn't save it as a real file, DVC has nothing to track — even though MLflow shows the run as successful.

### Running the Pipeline

```bash
python src/train.py        # run it manually first to confirm it works
ls models/model.pkl        # confirm the file exists
ls reports/metrics.json    # confirm the file exists

dvc repro                   # let DVC run the pipeline and register the outputs
dvc push                    # upload the new model/metrics to remote storage
git add .
git commit -m "Fix train.py to write model and metrics for DVC"
```

`dvc repro` is like saying "re-run anything that's out of date based on dependencies." If nothing changed, it does nothing — that's the efficiency DVC gives you over manually re-running everything.

### Step 3.6 — Simulating a Version Switch (The "Time Machine" Demo)

This step is designed to give beginners a "wow" moment.

1. **Change the dataset** (add a row, tweak a column).
2. **Re-track it**: `dvc add` creates a *new* pointer file pointing to the *new* data, tag it `data-v2`.
3. **Go back in time**: `git checkout data-v1` rewinds the pointer file to the old version, and `dvc checkout` swaps the actual dataset on disk to match.
4. **Re-run training**: `python src/train.py` now reproduces the *exact original results*, because the data, code, and parameters all match what they were back at `data-v1`.

This is the core promise of DVC: **Git controls which "version" you're pointing at; DVC swaps the actual files to match.**

---

## Common Errors and How to Read Them

Beginners often panic at these messages. Here's how to translate them:

| Error message | What it really means |
|----------------|------------------------|
| `WARNING: No file hash info found for 'models/model.pkl'` | DVC doesn't have a record that this file was ever produced through a tracked pipeline run. |
| `ERROR: Checkout failed for following targets: models/model.pkl` | DVC tried to restore this file from its cache, but the cache is empty — it was never pushed. |
| `FileNotFoundError: No such file or directory: 'data/raw/dataset.csv'` | The dataset pointer exists in Git, but the actual data was never pushed to DVC's storage, so `dvc checkout` couldn't restore it. |

### The Universal Fix Pattern

All of these errors boil down to one root cause: **a `.dvc` pointer file exists in Git, but the file it points to was never `dvc push`-ed to remote storage.**

The fix is always the same shape:

```bash
# 1. Make sure the source file actually exists locally
python src/generate_dataset.py

# 2. Re-register it with DVC and push to cache
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc
git commit -m "Re-track dataset with valid cache"
dvc push

# 3. Regenerate downstream outputs through the pipeline
dvc repro
dvc push
```

---

## The "Golden Rule" Workflow

To prevent all of the above errors from happening in the first place, beginners should memorize this sequence and follow it **every single time** they change code or data:

```bash
dvc repro       # 1. Regenerate any outdated outputs
dvc push        # 2. Save those outputs to remote storage
git add .       # 3. Stage the updated code/pointer files
git commit -m "your message"   # 4. Commit to Git
```

If you skip `dvc push`, your pointer files in Git will reference files that don't exist anywhere — a "broken link" waiting to bite you (or a teammate) later.

---

## Checkpoint — What Success Looks Like

By the end of Section 3, learners should be able to confirm:

- [ ] The dataset is tracked with `dvc add` and a `.dvc` pointer file is committed to Git
- [ ] The training pipeline runs end-to-end with `dvc repro`
- [ ] Both `models/model.pkl` and `reports/metrics.json` exist after a run
- [ ] They can switch between dataset versions (`data-v1` ↔ `data-v2`) and get matching, reproducible results

---

## Quick Glossary for Newbies

- **DVC remote**: External storage location (local folder, S3 bucket, etc.) where DVC keeps the actual large files.
- **`.dvc` file**: A small text "pointer" file that Git tracks instead of the real data/model file.
- **Pipeline stage**: A defined step (in `dvc.yaml`) with declared inputs, parameters, and expected outputs.
- **`dvc repro`**: Re-runs any pipeline stages whose inputs/parameters/code have changed since the last run.
- **`dvc push` / `dvc pull`**: Upload/download the actual data files to/from the DVC remote (like `git push`/`git pull`, but for big files).
- **`dvc checkout`**: Syncs the files in your working directory to match whatever version Git is currently pointing to.
