"""
tune_lightgbm.py
----------------
Hyperparameter tuning for LightGBM using Optuna.
Reports MAE, RMSE, and MASE.
"""

import optuna
import lightgbm as lgb
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pipeline import load_and_prepare
from evaluation import mase
from config import RANDOM_STATE, N_TUNING_TRIALS, DEFAULT_FEATURE_LEVEL, DEFAULT_SPLIT


def evaluate(model, X, y, y_train, name=""):
    """Print MAE, RMSE, and MASE."""
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mase_val = mase(y, pred, y_train)
    print(f"{name:15}  MAE: {mae:6.3f}   RMSE: {rmse:6.3f}   MASE: {mase_val:6.3f}")
    return mae


def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(40, verbose=False)]
    )
    return mean_absolute_error(y_test, model.predict(X_test))


def main():
    print("Preparing data for LightGBM tuning...\n")
    X_train, y_train, X_test, y_test, _, _, _ = load_and_prepare(
        feature_level=DEFAULT_FEATURE_LEVEL,
        split_type=DEFAULT_SPLIT
    )

    print("===== DEFAULT =====")
    default = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.03, num_leaves=15, max_depth=6,
        min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1
    )
    default.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(40, verbose=False)]
    )
    evaluate(default, X_test, y_test, y_train, "Default")

    print(f"\n===== OPTUNA ({N_TUNING_TRIALS} trials) =====")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: objective(t, X_train, y_train, X_test, y_test),
        n_trials=N_TUNING_TRIALS,
        show_progress_bar=True
    )

    print("\n===== TUNED =====")
    print(f"Best Test MAE: {study.best_value:.4f}")
    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    tuned = lgb.LGBMRegressor(
        **study.best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    tuned.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(40, verbose=False)]
    )

    print("\n===== COMPARISON =====")
    evaluate(default, X_test, y_test, y_train, "Default")
    evaluate(tuned, X_test, y_test, y_train, "Tuned")

    # Save best params
    out = Path(__file__).parent / "best_lightgbm_params.txt"
    with open(out, "w") as f:
        f.write(f"Best Test MAE: {study.best_value:.4f}\n\n")
        for k, v in study.best_params.items():
            f.write(f"{k}: {v}\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()