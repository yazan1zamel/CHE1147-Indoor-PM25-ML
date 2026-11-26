# 📊 Figure guide

| Figure | What it shows |
|---|---|
| [Model comparison](../Second_Model/figures/model_comparison.png) | Held-out MAE and RMSE for all five approaches |
| [Errors by concentration](../Second_Model/figures/errors_by_concentration.png) | Separate scales below and above 30 µg/m³ |
| [Temporal autocorrelation](../Second_Model/figures/temporal_autocorrelation.png) | Lag dependence for illustrative development households |
| [Correlation matrix](../Second_Model/figures/correlation_heatmap.png) | Pairwise correlation of key predictors and target |
| [LightGBM importance](../Second_Model/figures/lightgbm_importance.png) | Built-in split-count ranking |
| [XGBoost importance](../Second_Model/figures/xgboost_importance.png) | Normalized built-in gain ranking |
| [Observed versus predicted](../Second_Model/figures/xgboost_predicted_vs_actual.png) | XGBoost held-out comparison |
| [Example household](../Second_Model/figures/heldout_home_time_series.png) | Temporal illustration, not universal performance |

The built-in feature-importance measures are model-specific and must not be interpreted as causal contributions. Source files are in [`Second_Model/results/`](../Second_Model/results/).
