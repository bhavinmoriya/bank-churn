"""Gradio app. Run:  uv run python app.py"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bankchurn.features import engineer
from bankchurn.strategy import CampaignAssumptions, recommend_action, risk_tier, score_campaign

ART = Path("artifacts")
if not (ART / "model.pkl").exists():
    from bankchurn.train import main as train_main
    train_main()

BUNDLE = pickle.load(open(ART / "model.pkl", "rb"))
METRICS = json.loads((ART / "metrics.json").read_text())
TEST = pd.read_csv(ART / "test_customers.csv")
LR, GB, SC = BUNDLE["lr"], BUNDLE["gb"], BUNDLE["scaler"]
COLS = BUNDLE["columns"]


def _predict(df: pd.DataFrame):
    X = engineer(df)[COLS]
    Z = SC.transform(X.to_numpy())
    return LR.predict_proba(Z), GB.predict_proba(Z)[:, 1], Z


NICE = {"credit_score": "credit score", "age": "age", "age_over_45_sq": "age above 45", "tenure": "tenure",
        "log_balance": "balance level", "zero_balance": "zero balance", "high_balance": "balance > 100k",
        "log_salary": "salary", "has_credit_card": "credit card", "is_active_member": "active membership",
        "female": "gender (female)", "geo_Germany": "lives in Germany", "geo_Spain": "lives in Spain",
        "products_1": "only 1 product", "products_3": "3 products", "products_4": "4 products",
        "inactive_with_balance": "inactive with money on account"}


def predict_one(credit_score, geography, gender, age, tenure, balance, num_products, has_card, is_active, salary):
    row = dict(credit_score=credit_score, geography=geography, gender=gender, age=age, tenure=tenure,
               balance=balance, num_products=num_products, has_credit_card=int(has_card),
               is_active_member=int(is_active), estimated_salary=salary)
    df = pd.DataFrame([row])
    p_lr, p_gb, Z = _predict(df)
    p = float(p_lr[0])

    contrib = LR.w_ * Z[0]  # each feature's push on the log-odds (0 = average customer)
    order = np.argsort(contrib)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    top = np.r_[order[:4], order[-5:]]
    top = top[np.argsort(contrib[top])]
    ax.barh([NICE.get(COLS[i], COLS[i]) for i in top], contrib[top],
            color=["#2a9d8f" if contrib[i] < 0 else "#e76f51" for i in top])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("What pushes this customer toward leaving (red) or staying (green)", fontsize=9)
    ax.set_xlabel("effect on log-odds of churn vs. average customer")
    fig.tight_layout()

    drivers = [NICE.get(COLS[i], COLS[i]) for i in order[::-1][:3] if contrib[i] > 0]
    tier = risk_tier(p)
    verdict = f"## {tier} risk: {p:.0%} chance of leaving\n"
    verdict += f"*Gradient-boosting cross-check: {p_gb[0]:.0%}.*\n\n"
    verdict += "**Prediction:** " + ("likely to **leave**" if p >= 0.35 else "likely to **stay**") + "\n\n"
    if drivers:
        verdict += "**Main risk drivers:** " + ", ".join(drivers) + "\n\n"
    verdict += "**Suggested retention actions:**\n" + "\n".join(f"- {a}" for a in recommend_action(row, drivers))
    return verdict, fig


def run_campaign(offer_cost, success_rate, horizon):
    a = CampaignAssumptions(offer_cost, success_rate, horizon)
    p_lr, _, _ = _predict(TEST)
    s = score_campaign(TEST, p_lr, a)
    contacted = s[s.contact]
    total = contacted.net_value.sum()
    msg = (f"### Contact **{len(contacted):,}** of {len(s):,} customers ({len(contacted)/len(s):.0%})\n"
           f"- Expected net value of the campaign: **€{total:,.0f}**\n"
           f"- Budget needed: €{len(contacted) * a.offer_cost:,.0f}\n"
           f"- Expected customers saved: **{(contacted.p_churn * a.offer_success_rate).sum():.0f}**\n\n"
           "Notice: the target list is *not* simply the highest churn scores. High-value customers "
           "with moderate risk can beat low-value customers with high risk.")
    k = np.arange(0, len(s) + 1)
    cum = np.r_[0, np.cumsum(s.net_value.to_numpy())]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(k, cum)
    ax.axvline(len(contacted), ls="--", color="gray")
    ax.set_xlabel("customers contacted (ranked by net value)")
    ax.set_ylabel("cumulative net value (€)")
    ax.set_title("Where to stop spending")
    fig.tight_layout()
    table = contacted.head(25).round({"p_churn": 3, "annual_value": 0, "expected_gain": 0, "net_value": 0})
    return msg, fig, table


def model_report():
    rows = []
    for k, label in [("logreg_scratch", "Logistic regression (from scratch)"), ("gradient_boosting", "Gradient boosting")]:
        r = METRICS[k]
        rows.append([label, round(r["roc_auc"], 3), round(r["pr_auc"], 3), round(r["log_loss"], 3),
                     round(r["recall_at_0_35"], 2), round(r["precision_at_0_35"], 2), round(r["top_decile_lift"], 2)])
    perf = pd.DataFrame(rows, columns=["Model", "ROC-AUC", "PR-AUC", "Log-loss", "Recall@0.35", "Precision@0.35", "Top-decile lift"])
    dec = pd.DataFrame(METRICS["deciles_logreg"]).round(3)
    coefs = pd.Series(METRICS["coefficients"]).rename(index=NICE).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(coefs.index, coefs.values, color=["#2a9d8f" if v < 0 else "#e76f51" for v in coefs.values])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Logistic regression coefficients (standardised)")
    fig.tight_layout()
    note = (f"Overall churn rate **{METRICS['churn_rate']:.1%}**. A model predicting *nobody* churns would be "
            f"**{METRICS['baseline_accuracy_predict_nobody']:.1%} accurate** and catch zero churners, which is why we use AUC, "
            "recall and lift instead of accuracy.")
    return note, perf, dec, fig


with gr.Blocks(title="Bank Churn Lab") as demo:
    gr.Markdown("# 🏦 Bank Churn Lab\nPredict who will leave, understand why, and decide who is worth a retention offer. "
                "*Data is synthetic, generated with a known churn mechanism for learning purposes.*")
    with gr.Tab("1 · Single customer"):
        with gr.Row():
            with gr.Column():
                credit_score = gr.Slider(350, 850, 650, step=1, label="Credit score")
                geography = gr.Radio(["France", "Germany", "Spain"], value="France", label="Country")
                gender = gr.Radio(["Female", "Male"], value="Male", label="Gender")
                age = gr.Slider(18, 85, 42, step=1, label="Age")
                tenure = gr.Slider(0, 10, 4, step=1, label="Years with the bank")
                balance = gr.Number(value=80_000, label="Account balance (€)")
                num_products = gr.Radio([1, 2, 3, 4], value=1, label="Number of products")
                has_card = gr.Checkbox(True, label="Has credit card")
                is_active = gr.Checkbox(False, label="Active member")
                salary = gr.Number(value=60_000, label="Estimated salary (€)")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                verdict = gr.Markdown()
                plot = gr.Plot()
        inputs = [credit_score, geography, gender, age, tenure, balance, num_products, has_card, is_active, salary]
        btn.click(predict_one, inputs, [verdict, plot])
        demo.load(predict_one, inputs, [verdict, plot])
    with gr.Tab("2 · Retention campaign"):
        gr.Markdown("Set the economics. The app ranks held-out test customers by **expected net value** and tells you where to stop.")
        with gr.Row():
            oc = gr.Slider(10, 300, 60, step=5, label="Cost per offer (€)")
            sr = gr.Slider(0.05, 0.7, 0.30, step=0.05, label="Offer success rate")
            hz = gr.Slider(1, 5, 3, step=0.5, label="Value horizon (years)")
        go = gr.Button("Build target list", variant="primary")
        cmsg = gr.Markdown()
        cplot = gr.Plot()
        ctable = gr.Dataframe(label="Top 25 customers to contact")
        go.click(run_campaign, [oc, sr, hz], [cmsg, cplot, ctable])
        demo.load(run_campaign, [oc, sr, hz], [cmsg, cplot, ctable])
    with gr.Tab("3 · How good is the model?"):
        note = gr.Markdown()
        perf = gr.Dataframe(label="Held-out test performance")
        dec = gr.Dataframe(label="Decile table: top decile = riskiest 10% of customers")
        coefplot = gr.Plot()
        demo.load(model_report, None, [note, perf, dec, coefplot])

if __name__ == "__main__":
    demo.launch()
