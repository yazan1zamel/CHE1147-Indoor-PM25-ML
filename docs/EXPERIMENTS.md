# 🧭 Two experimental tracks — do not mix reported scores

| Item | Original team exploration | Second Model — paper-aligned |
|---|---|---|
| Entry points | [`main.ipynb`](../main.ipynb), [`pipeline.py`](../pipeline.py) | [`Second_Model/Second_Model_Executed.ipynb`](../Second_Model/Second_Model_Executed.ipynb) |
| Raw data | [`DATA/`](../DATA/) | Same unchanged [`DATA/`](../DATA/) |
| Model inputs | Original `get_reduced_features` setting | 35 selected predictors; prior completed intervals only |
| Household partition | Original seed 42; executed notebook reports 55/13 | Seed 2026; 54 development, 14 evaluation |
| Tuning | Separate Optuna tuning scripts | GroupKFold and randomized search within development households |
| Models compared | Linear, Ridge, LightGBM, XGBoost | Persistence, Linear, Ridge, LightGBM, XGBoost |
| Results | Original notebook/script outputs | [`Second_Model/results/`](../Second_Model/results/) and [`results_dashboard.html`](../Second_Model/results_dashboard.html) |

These tracks were created at different stages and have different validation procedures. **Do not combine their scores into one model-ranking table.**

**Code limitations in the original exploratory scripts:** `tune_lightgbm.py` and `tune_xgboost.py` use test labels during Optuna optimization; `nested_cv.py` uses training error inside its inner objective; and the original loader stacks files before resampling rather than joining sensors on household/time. Their results should be identified as exploratory.

**Second Model limitations:** PM₂.₅ target values were capped at 100 µg/m³; the task is next-10-minute prediction, not a one-hour forecast; these held-out homes were previously examined during other development runs; and the XGBoost model was trained in an additional separate run on the same paper-aligned partition. See the Second Model README.
