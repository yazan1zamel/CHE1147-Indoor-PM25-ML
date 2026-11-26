# 🧪 Methodology overview

## Data and target
Residential indoor PM₂.₅ data from the LBNL HENGH study; mean concentrations aggregated to ten-minute target intervals, with historical feature lags.

## Two workflows
- **Original team exploration:** unchanged root-level scripts and notebook, with the raw data in `DATA/`.
- **Second Model (paper-aligned):** [`Second_Model/`](../Second_Model/) includes the executed notebook, supplementary saved XGBoost experiment, numerical tables, and a self-contained HTML results dashboard.

## Second Model validation
68 eligible homes, 62,794 rows, 35 selected predictors. 54 development homes (50,690 rows), 14 evaluation homes (12,104 rows). A 5-fold `GroupKFold` selects models within the development partition using MAE. Persistence predicts the previous complete ten-minute PM₂.₅ mean.

**Limitations:** 100 µg/m³ concentration cap, retrospective use of previously inspected test homes, short prediction horizon, and short monitoring periods. There is no independent one-hour-ahead result in this folder. See [`docs/EXPERIMENTS.md`](EXPERIMENTS.md) for differences from the original scripts.
