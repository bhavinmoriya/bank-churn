"""Logistic regression from scratch (NumPy only).

Model:      p(churn | x) = sigmoid(w.x + b)
Loss:       mean binary cross-entropy + (lam/2)*||w||^2   (L2 / ridge)
Gradient:   dL/dw = X^T (p - y) / n + lam*w ;  dL/db = mean(p - y)
Optimiser:  full-batch gradient descent with momentum.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


class LogisticRegressionScratch:
    def __init__(self, lr: float = 0.3, n_iter: int = 1500, l2: float = 1e-3, momentum: float = 0.9):
        self.lr, self.n_iter, self.l2, self.momentum = lr, n_iter, l2, momentum
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        self.w_, self.b_ = np.zeros(d), 0.0
        vw, vb = np.zeros(d), 0.0
        eps = 1e-12
        for _ in range(self.n_iter):
            p = sigmoid(X @ self.w_ + self.b_)
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) \
                   + 0.5 * self.l2 * np.sum(self.w_ ** 2)
            self.loss_history_.append(float(loss))
            err = p - y
            gw = X.T @ err / n + self.l2 * self.w_
            gb = err.mean()
            vw = self.momentum * vw - self.lr * gw
            vb = self.momentum * vb - self.lr * gb
            self.w_ += vw
            self.b_ += vb
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.w_ + self.b_)
