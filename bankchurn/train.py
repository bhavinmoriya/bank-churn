"""Train + evaluate. Run:  uv run python -m bankchurn.train"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from bankchurn import metrics as M
from bankchurn.data import generate_customers
from bankchurn.features import Standardizer, engineer
from bankchurn.logreg import LogisticRegressionScratch

ART = Path("artifacts")


def main(n: int = 10_000, seed: int = 42):
    ART.mkdir(exist_ok=True)
    df = generate_customers(n, seed)
    tr, te = M.train_test_split_stratified(df["churned"].to_numpy(), 0.25, seed)
    X = engineer(df)
    y = df["churned"].to_numpy()

    scaler = Standardizer().fit(X.iloc[tr].to_numpy())
    Xtr, Xte = scaler.transform(X.iloc[tr].to_numpy()), scaler.transform(X.iloc[te].to_numpy())

    # 0) naive baseline: "nobody churns" -> shows why accuracy is misleading
    base_acc = 1 - y[te].mean()

    # 1) from-scratch logistic regression
    lr = LogisticRegressionScratch().fit(Xtr, y[tr])
    p_lr = lr.predict_proba(Xte)

    # 2) gradient boosting (sklearn) as a stronger non-linear benchmark
    gb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250, random_state=seed)
    gb.fit(Xtr, y[tr])
    p_gb = gb.predict_proba(Xte)[:, 1]

    report = {"churn_rate": float(y.mean()), "baseline_accuracy_predict_nobody": float(base_acc)}
    for name, p in [("logreg_scratch", p_lr), ("gradient_boosting", p_gb)]:
        prec, rec, f1 = M.precision_recall_f1(y[te], p, 0.35)
        report[name] = dict(roc_auc=M.roc_auc(y[te], p), pr_auc=M.average_precision(y[te], p),
                            log_loss=M.log_loss(y[te], p), precision_at_0_35=prec,
                            recall_at_0_35=rec, f1_at_0_35=f1,
                            top_decile_lift=M.decile_table(y[te], p)[0]["lift"])
    report["deciles_logreg"] = M.decile_table(y[te], p_lr)
    report["calibration_logreg"] = M.calibration_bins(y[te], p_lr)

    # global importance for the interpretable model: |standardised coefficient|
    report["coefficients"] = dict(zip(X.columns, map(float, lr.w_)))

    with open(ART / "model.pkl", "wb") as f:
        pickle.dump(dict(lr=lr, gb=gb, scaler=scaler, columns=list(X.columns)), f)
    df.iloc[te].reset_index(drop=True).to_csv(ART / "test_customers.csv", index=False)
    (ART / "metrics.json").write_text(json.dumps(report, indent=2))

    print(f"Churn rate: {y.mean():.1%} | 'nobody churns' accuracy: {base_acc:.1%}")
    for name in ["logreg_scratch", "gradient_boosting"]:
        r = report[name]
        print(f"{name:18s} ROC-AUC {r['roc_auc']:.3f} | PR-AUC {r['pr_auc']:.3f} | "
              f"logloss {r['log_loss']:.3f} | recall@0.35 {r['recall_at_0_35']:.2f} | "
              f"top-decile lift {r['top_decile_lift']:.2f}x")
    return report


if __name__ == "__main__":
    main()
