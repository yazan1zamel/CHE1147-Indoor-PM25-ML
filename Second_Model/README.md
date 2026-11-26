# 🧪 Second Model — Paper-Aligned 10-Minute Experiment

This directory contains the **separate revised experiment** supporting the term-paper model comparison. The team's original project files and original `DATA/` folder at the repository root are **unchanged**.

## 📖 Start here

| File | What it contains |
|---|---|
| [`results_dashboard.html`](results_dashboard.html) | **Self-contained HTML** summary with embedded charts; download and open in a browser, or use GitHub Pages if configured |
| [`Second_Model_Executed.ipynb`](Second_Model_Executed.ipynb) | **Executed** notebook with its saved outputs |
| [`Second_Model_Clean.ipynb`](Second_Model_Clean.ipynb) | Clean notebook without prior cell outputs, ready for a new run |
| [`executed_notebook.html`](executed_notebook.html) | HTML export of the executed notebook (including its output cells) |
| [`results/`](results/) | Model metrics, grouped CV, held-out predictions, per-home errors, XGBoost search, and feature importance |
| [`figures/`](figures/) | Curated report-ready images and supporting diagnostics |

## 🔍 Design

- **68 eligible households**, **62,794** ten-minute intervals.
- **54 development** households (50,690 observations); **14 evaluation** households (12,104 observations).
- **35 predictors**, from earlier completed intervals; indoor PM₂.₅ target capped at **100 µg/m³**.
- **Models:** persistence, linear regression, ridge, LightGBM, and XGBoost.
- **Tuning:** five-fold household GroupKFold on development data; model selection by MAE.

## 📊 Key held-out metrics

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| **Persistence** | **0.4173** | 2.6269 | 0.8712 |
| Linear regression | 0.5237 | 2.5918 | 0.8746 |
| Ridge regression | 0.5237 | 2.5918 | 0.8746 |
| LightGBM | 0.4832 | **2.4921** | **0.8840** |
| XGBoost | 0.4871 | 2.5373 | 0.8798 |

[📈 View figure gallery](../assets/figures/) · [📝 Read experiment differences](../docs/EXPERIMENTS.md)

## ⚠️ Important

The **executed notebook does not contain the separate later XGBoost run**; the corresponding CSV results and plots are in [`results/`](results/). The combined HTML dashboard integrates those saved outputs.

These are retrospective held-out results, not a fully untouched external test: the households were inspected during earlier project work. They are **next-10-minute forecasts**, not one-hour-ahead results.

### Re-running

Install the dependencies from [`requirements-paper.in`](../requirements-paper.in). Open `Second_Model_Clean.ipynb`, set the notebook's `USER_DATA_ROOT` to the repository's **`DATA`** directory (or your local download), then execute its cells. The original code contains a Mac-specific fallback path, so adapt that setting on another machine. **XGBoost was separately tuned**, using the same group split and predictor set; the saved output is provided here.

**View HTML:** GitHub shows HTML source/download, not a live rendered app. Download `results_dashboard.html` and double-click it to view offline. To serve it from the repository online, use GitHub Pages where permitted.
