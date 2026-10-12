"""Evaluation metrics written from scratch - because accuracy lies on imbalanced data.

If 80% of customers stay, a model that says "nobody leaves" is 80% accurate and
completely useless. We need metrics that care about *ranking* the leavers.
"""
from __future__ import annotations

import numpy as np


def train_test_split_stratified(n_or_y, test_size=0.25, seed=0):
    y = np.asarray(n_or_y)
    rng = np.random.default_rng(seed)
    test_idx = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        test_idx.extend(idx[: int(round(len(idx) * test_size))])
    test_mask = np.zeros(len(y), bool)
    test_mask[test_idx] = True
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def roc_auc(y, p):
    """P(random churner scores higher than random stayer) via rank statistics."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    # average ranks for ties
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = y.sum(), (1 - y).sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def average_precision(y, p):
    """Area under the precision-recall curve (step-wise)."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p), kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(ys) + 1)
    return float((precision * ys).sum() / ys.sum())


def confusion(y, p, threshold=0.5):
    pred = (np.asarray(p) >= threshold).astype(int)
    y = np.asarray(y)
    return dict(tp=int(((pred == 1) & (y == 1)).sum()), fp=int(((pred == 1) & (y == 0)).sum()),
                fn=int(((pred == 0) & (y == 1)).sum()), tn=int(((pred == 0) & (y == 0)).sum()))


def precision_recall_f1(y, p, threshold=0.5):
    c = confusion(y, p, threshold)
    prec = c["tp"] / max(c["tp"] + c["fp"], 1)
    rec = c["tp"] / max(c["tp"] + c["fn"], 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return prec, rec, f1


def log_loss(y, p, eps=1e-12):
    y, p = np.asarray(y), np.clip(np.asarray(p), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def decile_table(y, p, n_bins=10):
    """Sort customers by risk, cut into deciles; compare churn rate vs. average ('lift')."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p))
    ys = y[order]
    base = y.mean()
    rows = []
    for k, chunk in enumerate(np.array_split(ys, n_bins), start=1):
        rows.append(dict(decile=k, customers=len(chunk), churners=int(chunk.sum()),
                         churn_rate=float(chunk.mean()), lift=float(chunk.mean() / base)))
    return rows


def calibration_bins(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            out.append((float(p[m].mean()), float(y[m].mean())))
    return out
