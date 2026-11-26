"""
nested_cv.py
------------
Nested cross-validation for unbiased performance estimation.
Outer loop: GroupKFold by household
Inner loop: Optuna hyperparameter tuning
"""

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_household_files, resample_data
from feature_engineering import (
    add_lag_features, add_time_features, get_reduced_features
)
from evaluation import mase
from config import (
    DATA_ROOT, TARGET, UPPER_LIMIT, LAG_STEPS, LAG_COLUMNS,
    RANDOM_STATE, N_CV_SPLITS, DEFAULT_FEATURE_LEVEL, N_TUNING_TRIALS
)


def prepare_full_data(root=None):
    """Load and prepare the full dataset."""
    if root is None:
        root = DATA_ROOT

    df = load_household_files(root)
    df = resample_data(df, freq="10min")
    df[TARGET] = df[TARGET].clip(upper=UPPER_LIMIT)
    df = add_lag_features(df, LAG_COLUMNS, LAG_STEPS)
    df = add_time_features(df)
    df = df.dropna(subset=[TARGET, "PM_MET_IN1_lag1"]).reset_index(drop=True)
    return df


def _prepare_fold(train, test, feature_level):
    """Prepare features for one outer fold (no leakage)."""
    hh_means = train.groupby("household_id")[TARGET].mean()
    train = train.copy()
    test = test.copy()
    train["hh_mean_pm"] = train["household_id"].map(hh_means)
    test["hh_mean_pm"] = test["household_id"].map(hh_means)
    test["hh_mean_pm"] = test["hh_mean_pm"].fillna(train[TARGET].mean())

    feature_cols = get_reduced_features(train.columns.tolist(), level=feature_level)
    X_train = train[feature_cols]
    X_test = test[feature_cols]
    valid_cols = [c for c in feature_cols if X_train[c].notna().any()]
    X_train = X_train[valid_cols]
    X_test = X_test[valid_cols]

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=valid_cols)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=valid_cols)

    return X_train, train[TARGET], X_test, test[TARGET]


def nested_cv(model_type="lightgbm", n_outer=N_CV_SPLITS, n_trials=15, feature_level=DEFAULT_FEATURE_LEVEL):
    """
    Run nested cross-validation.
    
    Parameters
    ----------
    model_type : str
        "lightgbm" or "xgboost"
    n_outer : int
        Number of outer folds
    n_trials : int
        Optuna trials per outer fold
    """
    df = prepare_full_data()
    groups = df["household_id"]
    outer_cv = GroupKFold(n_splits=n_outer)

    outer_mae, outer_rmse, outer_mase = [], [], []

    print(f"Nested CV | Model: {model_type} | Outer folds: {n_outer} | Trials: {n_trials}\n")

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(df, groups=groups), 1):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        X_train, y_train, X_test, y_test = _prepare_fold(train, test, feature_level)

        # Inner loop: Optuna
        def objective(trial):
            if model_type == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 8, 48),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1,
                    "verbose": -1
                }
                model = lgb.LGBMRegressor(**params)
            else:  # xgboost
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "min_child_weight": trial.suggest_int("min_child_weight", 5, 40),
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1,
                    "verbosity": 0
                }
                model = xgb.XGBRegressor(**params)

            model.fit(X_train, y_train)
            return mean_absolute_error(y_train, model.predict(X_train))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Train final model with best params on outer training set
        if model_type == "lightgbm":
            best_model = lgb.LGBMRegressor(**study.best_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        else:
            best_model = xgb.XGBRegressor(**study.best_params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)

        best_model.fit(X_train, y_train)
        pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mase_val = mase(y_test, pred, y_train)

        outer_mae.append(mae)
        outer_rmse.append(rmse)
        outer_mase.append(mase_val)

        print(f"Outer fold {fold}: MAE={mae:.3f}  RMSE={rmse:.3f}  MASE={mase_val:.3f}")

    print("\n" + "=" * 50)
    print(f"NESTED CV SUMMARY ({model_type.upper()})")
    print("=" * 50)
    print(f"MAE  : {np.mean(outer_mae):.3f} ± {np.std(outer_mae):.3f}")
    print(f"RMSE : {np.mean(outer_rmse):.3f} ± {np.std(outer_rmse):.3f}")
    print(f"MASE : {np.mean(outer_mase):.3f} ± {np.std(outer_mase):.3f}")

    return {"mae": outer_mae, "rmse": outer_rmse, "mase": outer_mase}


def main():
    print("Running nested CV for LightGBM...")
    nested_cv(model_type="lightgbm", n_trials=15)

    print("\n" + "-" * 50 + "\n")

    print("Running nested CV for XGBoost...")
    nested_cv(model_type="xgboost", n_trials=15)


if __name__ == "__main__":
    main()