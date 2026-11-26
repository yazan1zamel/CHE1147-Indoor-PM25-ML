"""
cross_validation.py
-------------------
GroupKFold cross-validation by household.
Reports MAE, RMSE, and MASE.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb

from data_loader import load_household_files, resample_data
from feature_engineering import (
    add_lag_features, add_time_features, get_reduced_features
)
from evaluation import mase
from config import (
    DATA_ROOT, TARGET, UPPER_LIMIT, LAG_STEPS, LAG_COLUMNS,
    DEFAULT_N_ESTIMATORS, DEFAULT_LEARNING_RATE, DEFAULT_MAX_DEPTH,
    DEFAULT_NUM_LEAVES, RANDOM_STATE, N_CV_SPLITS, DEFAULT_FEATURE_LEVEL
)


def prepare_full_data(root=None):
    """Load and prepare the full dataset for cross-validation."""
    if root is None:
        root = DATA_ROOT

    df = load_household_files(root)
    df = resample_data(df, freq="10min")
    df[TARGET] = df[TARGET].clip(upper=UPPER_LIMIT)
    df = add_lag_features(df, LAG_COLUMNS, LAG_STEPS)
    df = add_time_features(df)
    df = df.dropna(subset=[TARGET, "PM_MET_IN1_lag1"]).reset_index(drop=True)
    return df


def run_cv(df, n_splits=N_CV_SPLITS, feature_level=DEFAULT_FEATURE_LEVEL):
    """
    Run GroupKFold CV by household.
    Returns dictionary of MAE, RMSE, and MASE for each model.
    """
    groups = df["household_id"]
    gkf = GroupKFold(n_splits=n_splits)

    results = {
        "Linear": {"mae": [], "rmse": [], "mase": []},
        "Ridge": {"mae": [], "rmse": [], "mase": []},
        "LightGBM": {"mae": [], "rmse": [], "mase": []},
        "XGBoost": {"mae": [], "rmse": [], "mase": []}
    }

    print(f"Running {n_splits}-fold GroupKFold (feature level: {feature_level})\n")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), 1):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        # Household mean from training fold only
        hh_means = train.groupby("household_id")[TARGET].mean()
        train["hh_mean_pm"] = train["household_id"].map(hh_means)
        test["hh_mean_pm"] = test["household_id"].map(hh_means)
        test["hh_mean_pm"] = test["hh_mean_pm"].fillna(train[TARGET].mean())

        # Features
        feature_cols = get_reduced_features(train.columns.tolist(), level=feature_level)
        X_train = train[feature_cols]
        X_test = test[feature_cols]
        valid_cols = [c for c in feature_cols if X_train[c].notna().any()]
        X_train = X_train[valid_cols]
        X_test = X_test[valid_cols]

        # Impute
        imputer = SimpleImputer(strategy="median")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=valid_cols)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=valid_cols)

        y_train = train[TARGET]
        y_test = test[TARGET]

        # ---------- Linear ----------
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
        lr.fit(X_train, y_train)
        pred = lr.predict(X_test)
        results["Linear"]["mae"].append(mean_absolute_error(y_test, pred))
        results["Linear"]["rmse"].append(np.sqrt(mean_squared_error(y_test, pred)))
        results["Linear"]["mase"].append(mase(y_test, pred, y_train))

        # ---------- Ridge ----------
        ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0))
        ])
        ridge.fit(X_train, y_train)
        pred = ridge.predict(X_test)
        results["Ridge"]["mae"].append(mean_absolute_error(y_test, pred))
        results["Ridge"]["rmse"].append(np.sqrt(mean_squared_error(y_test, pred)))
        results["Ridge"]["mase"].append(mase(y_test, pred, y_train))

        # ---------- LightGBM ----------
        lgbm = lgb.LGBMRegressor(
            n_estimators=DEFAULT_N_ESTIMATORS,
            learning_rate=DEFAULT_LEARNING_RATE,
            num_leaves=DEFAULT_NUM_LEAVES,
            max_depth=DEFAULT_MAX_DEPTH,
            min_child_samples=30,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        lgbm.fit(X_train, y_train)
        pred = lgbm.predict(X_test)
        results["LightGBM"]["mae"].append(mean_absolute_error(y_test, pred))
        results["LightGBM"]["rmse"].append(np.sqrt(mean_squared_error(y_test, pred)))
        results["LightGBM"]["mase"].append(mase(y_test, pred, y_train))

        # ---------- XGBoost ----------
        xgb_model = xgb.XGBRegressor(
            n_estimators=DEFAULT_N_ESTIMATORS,
            learning_rate=DEFAULT_LEARNING_RATE,
            max_depth=DEFAULT_MAX_DEPTH,
            min_child_weight=30,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        pred = xgb_model.predict(X_test)
        results["XGBoost"]["mae"].append(mean_absolute_error(y_test, pred))
        results["XGBoost"]["rmse"].append(np.sqrt(mean_squared_error(y_test, pred)))
        results["XGBoost"]["mase"].append(mase(y_test, pred, y_train))

        print(f"Fold {fold}: "
              f"Linear MAE={results['Linear']['mae'][-1]:.3f} | "
              f"XGB MAE={results['XGBoost']['mae'][-1]:.3f}")

    return results


def print_summary(results):
    """Print mean ± std for MAE, RMSE, and MASE."""
    print("\n" + "=" * 65)
    print("CROSS-VALIDATION SUMMARY (GroupKFold by Household)")
    print("=" * 65)
    for model_name, scores in results.items():
        print(f"\n{model_name}:")
        print(f"  MAE  : {np.mean(scores['mae']):.3f} ± {np.std(scores['mae']):.3f}")
        print(f"  RMSE : {np.mean(scores['rmse']):.3f} ± {np.std(scores['rmse']):.3f}")
        print(f"  MASE : {np.mean(scores['mase']):.3f} ± {np.std(scores['mase']):.3f}")


def main():
    print(f"Using data folder: {DATA_ROOT.resolve()}\n")
    df = prepare_full_data()
    print(f"Total rows: {len(df):,}")
    print(f"Households: {df['household_id'].nunique()}\n")

    results = run_cv(df)
    print_summary(results)


if __name__ == "__main__":
    main()