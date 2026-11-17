# Predicting 30-day readmission of diabetic patients — 7CS133 coursework (Problem A)

A hospital wants to know, on the day a diabetic patient goes home, how likely they are to come back
within 30 days. This code builds and compares four models, makes their risk scores honest, picks a
cost-based cut-off, explains the model with SHAP, and checks it for fairness.

## What is in this folder

```
README.md            ← you are here
requirements.txt     ← library versions
data/                ← put the two CSV files here (NOT included / NOT uploaded)
notebooks/
  01_data_and_features.ipynb            Steps 1–2: load, clean, new features, split by patient
  02_models_and_tuning.ipynb            Step 3:    logistic regression, random forest, LightGBM (tuned)
  03_mlp_embeddings.ipynb               Step 4:    neural network with embeddings
  04_evaluation_explain_fairness.ipynb  Steps 5–8: test results, calibration, threshold, SHAP, fairness
results/             ← files passed from one notebook to the next (created when you run)
```

`results/` holds only what the next notebook needs:

| File | Made by | Used by |
|---|---|---|
| `prepared_data.joblib` | 01 | 02, 03, 04 |
| `cv_results.csv` | 02 | 04 |
| `final_models.joblib` | 02 | 04 |
| `mlp_predictions.joblib` | 03 | 04 |

All tables and figures are shown inside the notebooks themselves.

## How to run it — 4 easy steps

**1. Get the data.**
Download the zip from <https://archive.ics.uci.edu/dataset/296>, unzip it, and put
`diabetic_data.csv` and `IDs_mapping.csv` in the `data/` folder.
(The zip calls the second file `IDS_mapping.csv`; either spelling works.)

**2. Install the libraries** (once). In a terminal, inside this folder:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

On Google Colab, skip this and run this once in a new cell at the top of each notebook:
`!pip install -q lightgbm shap fairlearn`

**Mac only:** LightGBM needs one extra system library. If you see `Library not loaded: libomp.dylib`, run this once (needs [Homebrew](https://brew.sh)):

```bash
brew install libomp
```

Tested with Python 3.11 and the versions in `requirements.txt`.

**3. Start Jupyter.**

```bash
jupyter notebook
```

**4. Run the notebooks in order: 01 → 02 → 03 → 04.**
In each one, choose *Kernel → Restart & Run All*. Each notebook saves its results for the next one.

| Notebook | Roughly how long |
|---|---|
| 01 data and features | 1 minute |
| 02 models and tuning | 15–30 minutes (random forest is the slow part) |
| 03 MLP with embeddings | 5–10 minutes |
| 04 evaluation, SHAP, fairness | 3–5 minutes |

Short of time? Lower `n_iter` in notebook 02 (Steps 3c and 3d).

## How each notebook is laid out

Each step starts with a short explanation in plain language, often with an everyday comparison, then the code,
then what the result shows. The notebooks are written as a walk-through of the work, so they can be read from top
to bottom like a short story of the project.

## Good to know

- All random seeds are fixed (`SEED = 42`), so you get the same numbers every run.
- Every split is done **by patient** (`patient_nbr`), so no patient is in two parts (no data leakage).
- The TEST set is only used once, in notebook 04.
- The final model is LightGBM (chosen in notebook 04, Step 5d, variable `BEST`).
- The data and `results/` are not part of the submission; running the notebooks recreates `results/`.
