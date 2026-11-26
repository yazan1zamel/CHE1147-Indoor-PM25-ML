"""
pipeline.py
-----------
Shared data loading and preparation.
"""

import pandas as pd
from sklearn.impute import SimpleImputer

from config import (
    DATA_ROOT, TARGET, UPPER_LIMIT, DEFAULT_FREQ,
    LAG_STEPS, LAG_COLUMNS, RANDOM_STATE, DEFAULT_TEST_FRACTION
)
from data_loader import load_household_files, resample_data
from feature_engineering import (
    add_lag_features, add_time_features, add_household_mean,
    household_split, get_reduced_features
)


def load_and_prepare(
    freq: str = DEFAULT_FREQ,
    feature_level: str = "medium",
    split_type: str = "household",
    test_fraction: float = DEFAULT_TEST_FRACTION,
    root=None
):
    """Load, resample, create features, split, and impute."""
    if root is None:
        root = DATA_ROOT

    print(f"Loading data | freq={freq} | features={feature_level} | split={split_type}")

    df = load_household_files(root)
    df = resample_data(df, freq=freq)
    df[TARGET] = df[TARGET].clip(upper=UPPER_LIMIT)

    df = add_lag_features(df, LAG_COLUMNS, LAG_STEPS)
    df = add_time_features(df)
    df = df.dropna(subset=[TARGET, "PM_MET_IN1_lag1"]).reset_index(drop=True)

    if split_type == "household":
        train, test = household_split(df, test_fraction=test_fraction, random_state=RANDOM_STATE)
    else:
        df = df.sort_values("timestamp").reset_index(drop=True)
        cutoff = df["timestamp"].quantile(1 - test_fraction)
        train = df[df["timestamp"] < cutoff].copy()
        test = df[df["timestamp"] >= cutoff].copy()

    train, test = add_household_mean(train, test, TARGET)

    feature_cols = get_reduced_features(train.columns.tolist(), level=feature_level)

    X_train = train[feature_cols]
    X_test = test[feature_cols]
    valid_cols = [c for c in feature_cols if X_train[c].notna().any()]
    X_train = X_train[valid_cols]
    X_test = X_test[valid_cols]
    feature_cols = valid_cols

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

    y_train = train[TARGET]
    y_test = test[TARGET]

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {len(feature_cols)}")
    return X_train, y_train, X_test, y_test, feature_cols, train, test