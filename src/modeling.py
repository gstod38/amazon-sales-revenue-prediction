from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


@dataclass
class ModelBundle:
    name: str
    model: object
    y_pred: np.ndarray


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def train_hist_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def predict(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def feature_importance(model, feature_names) -> pd.Series | None:
    """
    Returns feature importance if available, else None.
    Works for tree models that expose `feature_importances_`.
    """
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        return importances.sort_values(ascending=False)
    return None
