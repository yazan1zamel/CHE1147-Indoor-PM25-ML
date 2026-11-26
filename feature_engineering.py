"""
feature_engineering.py
----------------------
This file creates the features (predictors) that the models will use.
It focuses on lag features, time features, and household-level information.
"""

import pandas as pd
import numpy as np
from config import (
    RANDOM_STATE, DEFAULT_TEST_FRACTION, DEFAULT_FEATURE_LEVEL
)

def add_lag_features(df: pd.DataFrame, columns: list, lag_steps: list) -> pd.DataFrame:
    """
    Create lag features: the value of a sensor from previous time steps.
    
    Example: PM_MET_IN1_lag1 = indoor PM from 1 step ago
             (10 minutes ago if using 10-min data)
    
    These are the most important features for predicting future PM.
    """
    df = df.sort_values(["household_id", "timestamp"]).copy()
    for col in columns:
        if col in df.columns:
            for lag in lag_steps:
                df[f"{col}_lag{lag}"] = df.groupby("household_id")[col].shift(lag)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple time-based features (hour of day, day of week, etc.)."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["month"] = df["timestamp"].dt.month
    return df


def add_household_mean(train: pd.DataFrame, test: pd.DataFrame, target: str) -> tuple:
    """
    Add the average PM level of each household.
    
    Important: the average is calculated only from the training period
    to avoid data leakage.
    """
    means = train.groupby("household_id")[target].mean()
    train = train.copy()
    test = test.copy()
    train["hh_mean_pm"] = train["household_id"].map(means)
    test["hh_mean_pm"] = test["household_id"].map(means)
    test["hh_mean_pm"] = test["hh_mean_pm"].fillna(train[target].mean())
    return train, test

def household_split(df: pd.DataFrame, test_fraction: float = DEFAULT_TEST_FRACTION, random_state: int = RANDOM_STATE):
    """
    Split by household so the model is tested on homes it has never seen.
    """
    households = df["household_id"].unique()
    np.random.seed(random_state)
    np.random.shuffle(households)

    n_test = max(1, int(len(households) * test_fraction))
    test_homes = households[:n_test]
    train_homes = households[n_test:]

    train = df[df["household_id"].isin(train_homes)].copy()
    test = df[df["household_id"].isin(test_homes)].copy()

    print(f"Household split: {len(train_homes)} train homes, {len(test_homes)} test homes")
    print(f"Train rows: {len(train):,} | Test rows: {len(test):,}")
    return train, test

def time_based_split(df: pd.DataFrame, test_fraction: float = DEFAULT_TEST_FRACTION):
    """
    Split the data chronologically so the model is tested on future data.
    
    This avoids temporal leakage (using future information to predict the past).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = df["timestamp"].quantile(1 - test_fraction)
    train = df[df["timestamp"] < cutoff].copy()
    test = df[df["timestamp"] >= cutoff].copy()
    return train, test

def get_reduced_features(available_columns: list, level: str = DEFAULT_FEATURE_LEVEL) -> list:
    """Return reduced feature set. Levels: ultra_low, low, medium, high."""
    ultra_low = ["PM_MET_IN1_lag1", "PM_MET_IN1_lag2", "PM_MET_OUT", "hh_mean_pm", "CO2_ETC_IN1"]
    low = ultra_low + ["PM_MET_IN1_lag3", "QCFM_Exhaust", "ACH", "hour"]
    medium = low + ["PM_MET_IN1_lag6", "PM_MET_IN1_lag12", "T_ETC_IN1", "RH_ETC_IN1",
                    "T_ETC_OUT", "RH_ETC_OUT", "QCFM_Total", "ACH_lag1"]
    high = medium + ["PM_MET_IN1_lag36", "PM_MET_IN1_lag72", "dayofweek", "is_weekend", "month"]

    mapping = {"ultra_low": ultra_low, "low": low, "medium": medium, "high": high}
    if level not in mapping:
        raise ValueError("level must be ultra_low, low, medium, or high")
    return [f for f in mapping[level] if f in available_columns]