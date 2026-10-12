"""Feature engineering + a tiny from-scratch standardizer.

Logistic regression is *linear in its inputs*, so we help it by giving it
non-linear features (age^2, product dummies, 'zero balance' flag...).
Tree models find these shapes on their own - a nice comparison later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["credit_score"] = df["credit_score"]
    X["age"] = df["age"]
    X["age_over_45_sq"] = np.maximum(df["age"] - 45, 0) ** 2
    X["tenure"] = df["tenure"]
    X["log_balance"] = np.log1p(df["balance"])
    X["zero_balance"] = (df["balance"] == 0).astype(int)
    X["high_balance"] = (df["balance"] > 100_000).astype(int)
    X["log_salary"] = np.log(df["estimated_salary"])
    X["has_credit_card"] = df["has_credit_card"]
    X["is_active_member"] = df["is_active_member"]
    X["female"] = (df["gender"] == "Female").astype(int)
    for g in ["Germany", "Spain"]:           # France = reference group
        X[f"geo_{g}"] = (df["geography"] == g).astype(int)
    for k in [1, 3, 4]:                       # 2 products = reference group
        X[f"products_{k}"] = (df["num_products"] == k).astype(int)
    X["inactive_with_balance"] = ((df["is_active_member"] == 0) & (df["balance"] > 0)).astype(int)
    return X.astype(float)


class Standardizer:
    """z = (x - mean) / std, fit on the training set only (no leakage!)."""

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_
