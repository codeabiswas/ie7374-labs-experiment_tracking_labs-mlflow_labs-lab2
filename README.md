# Breast Cancer Detection Lab — MLflow Experiment Tracking

A machine learning lab demonstrating the full model lifecycle using **MLflow**: experiment tracking, hyperparameter tuning, model registry, and serving. Predicts whether a tumour is malignant or benign using the UCI Breast Cancer dataset.

## Changes from Original

The original lab used the UCI Wine Quality dataset with XGBoost. This version swaps both:

| | Original | This version |
|---|---|---|
| **Dataset** | Wine Quality (CSV files, 6,497 samples, 12 features) | Breast Cancer (`sklearn` built-in, 569 samples, 30 features) |
| **Advanced model** | XGBoost | LightGBM |
| **Hyperparameter** | `max_depth` (depth-wise trees) | `num_leaves` (leaf-wise trees) |
| **Early stopping** | `early_stopping_rounds=` param | `lgb.early_stopping()` callback |
| **Autologging** | `mlflow.xgboost.autolog()` | `mlflow.lightgbm.autolog()` |
| **Model name** | `wine_quality` | `cancer_detection` |
| **MLflow parent run** | `xgboost_models` | `lgbm_models` |
| **Data splits** | No stratification | `stratify=y` on both splits |

The Random Forest baseline, all MLflow registry/serving patterns, and the Hyperopt + SparkTrials sweep structure are unchanged.

## Project Description

The notebook (`starter.ipynb`) walks through the complete ML lifecycle in 18 steps:

1. **Data loading** — `load_breast_cancer()` from scikit-learn; 30 numeric tumour features, binary target (0 = malignant, 1 = benign)
2. **EDA** — class distribution, summary statistics, box plots of key features
3. **Preprocessing** — stratified 60/20/20 train/val/test split
4. **Baseline model** — `RandomForestClassifier(n_estimators=10)` wrapped in a custom `mlflow.pyfunc.PythonModel` to expose probabilities; logged to MLflow
5. **Model registry** — baseline registered as `cancer_detection` v1 and promoted to Production
6. **Hyperparameter tuning** — LightGBM trained across 96 configurations via Hyperopt TPE + SparkTrials; each trial is a nested MLflow run
7. **Best model promotion** — highest-AUC LightGBM registered as `cancer_detection` v2, promoted to Production; v1 archived
8. **Batch inference** — Spark UDF wrapping the Production model
9. **Real-time serving** — `mlflow models serve` exposes a REST endpoint

## How to Run

### Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- Java 8+ (required by PySpark)

### Setup

```bash
# Create a virtual environment and install dependencies
uv venv --python 3.11
uv pip install -r requirements.txt
```

### Run the notebook

```bash
uv run jupyter notebook starter.ipynb
```

Run all cells top-to-bottom. The notebook is self-contained — no external data files are needed.

### View the MLflow UI

In a separate terminal:

```bash
uv run mlflow ui --port 5001
```

Then open [http://localhost:5001](http://localhost:5001) to browse experiments, runs, and the model registry.

### Real-time serving

After the notebook has registered and promoted the LightGBM model:

```bash
uv run mlflow models serve --env-manager=local -m models:/cancer_detection/production -h 0.0.0.0 -p 5001
```

Send a prediction request:

```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

## Expected Results

| Model | Expected AUC |
|-------|-------------|
| Random Forest baseline | ~0.97 |
| LightGBM (tuned) | ~0.99 |

## Dependencies

See `requirements.txt`. Key packages: `mlflow`, `lightgbm`, `scikit-learn`, `hyperopt`, `pyspark`.
