from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    results = {
      "Model Name": {"MAE": ..., "RMSE": ..., "R2": ...},
      ...
    }
    """
    df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    # nice ordering
    df = df[["Model", "MAE", "RMSE", "R2"]]
    return df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)


def plot_residuals(y_true, y_pred, title: str = "Residual Plot") -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 4))
    plt.scatter(y_pred, residuals, alpha=0.2)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(title)
    plt.show()


def plot_actual_vs_pred(y_true, y_pred, title: str = "Actual vs Predicted") -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.2)
    min_v = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_v = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.show()
