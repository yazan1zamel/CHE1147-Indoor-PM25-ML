"""
evaluation.py
-------------
Metrics and model comparison.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mase(y_true, y_pred, y_train):
    """Mean Absolute Scaled Error vs naive lag-1 baseline."""
    mae_model = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    naive_errors = np.abs(np.diff(y_train))
    mae_naive = np.mean(naive_errors)
    return mae_model / mae_naive if mae_naive != 0 else np.nan


def evaluate(model, X, y, y_train_for_mase, name=""):
    """Print MAE, RMSE, and MASE."""
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mase_val = mase(y, pred, y_train_for_mase)
    print(f"{name:15}  MAE: {mae:6.3f}   RMSE: {rmse:6.3f}   MASE: {mase_val:6.3f}")
    return pred, mae, rmse, mase_val


def compare_models(models: dict, X_train, y_train, X_test, y_test):
    """Compare models on both training and test sets."""
    print("=== Training Set ===")
    for name, model in models.items():
        evaluate(model, X_train, y_train, y_train, name)

    print("\n=== Test Set ===")
    results = []
    for name, model in models.items():
        _, mae, rmse, mase_val = evaluate(model, X_test, y_test, y_train, name)
        results.append({
            "Model": name,
            "Test MAE": round(mae, 3),
            "Test RMSE": round(rmse, 3),
            "Test MASE": round(mase_val, 3)
        })
    return results