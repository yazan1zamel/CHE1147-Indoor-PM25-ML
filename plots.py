"""
plots.py
--------
All plotting functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

from config import PLOTS_DIR


def plot_target_distribution(y_train, y_test):
    plt.figure(figsize=(8, 4))
    plt.hist(y_train, bins=50, alpha=0.7, label="Train")
    plt.hist(y_test, bins=50, alpha=0.7, label="Test")
    plt.xlabel("Indoor PM2.5 (µg/m³)")
    plt.title("Target Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "target_distribution.png", dpi=150)
    plt.show()


def plot_top_correlations(X_train, y_train, top_n=10):
    corr = X_train.corrwith(y_train).abs().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, 4))
    plt.barh(corr.index[::-1], corr.values[::-1])
    plt.xlabel("Absolute Correlation")
    plt.title("Top Features Correlated with Target")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_correlations.png", dpi=150)
    plt.show()


def plot_pred_vs_actual(y_true, y_pred, title="Predicted vs Actual"):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=8)
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], "r--", lw=2)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pred_vs_actual.png", dpi=150)
    plt.show()


def plot_feature_importance(model, feature_names, title="Top Features", top_n=10):
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, 5))
    plt.barh(importance["feature"][::-1], importance["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
    plt.show()


def plot_household_examples(test_df, y_true, y_pred, n_best=2, n_worst=2):
    """Plot best and worst fitting households."""
    results = test_df.copy()
    results["actual"] = y_true.values if hasattr(y_true, "values") else y_true
    results["predicted"] = y_pred

    hh_mae = (
        results.groupby("household_id")
        .apply(lambda g: mean_absolute_error(g["actual"], g["predicted"]))
        .sort_values()
    )

    best = hh_mae.head(n_best).index.tolist()
    worst = hh_mae.tail(n_worst).index.tolist()

    def _plot_one(hid, label):
        hh = results[results["household_id"] == hid].sort_values("timestamp")
        plt.figure(figsize=(12, 4))
        plt.plot(hh["timestamp"], hh["actual"], label="Measured")
        plt.plot(hh["timestamp"], hh["predicted"], label="Predicted", alpha=0.8)
        plt.title(f"Household {hid} ({label})")
        plt.xlabel("Time")
        plt.ylabel("PM2.5")
        plt.legend()
        plt.tight_layout()
        plt.show()

    for hid in best:
        _plot_one(hid, "Best fit")
    for hid in worst:
        _plot_one(hid, "Worst fit")

    return best, worst