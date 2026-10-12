"""From prediction to *decision*: who should we spend retention money on?

A churn score alone does not save customers. A retention campaign costs money,
only some customers respond, and customers differ in how much they are worth.
So we rank by expected net value, not by churn probability alone:

    expected_gain = P(churn) * P(offer works) * customer_value_over_horizon
    net_value     = expected_gain - offer_cost

Contact a customer only if net_value > 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CampaignAssumptions:
    offer_cost: float = 60.0         # EUR per contacted customer (call, bonus, fee waiver)
    offer_success_rate: float = 0.30 # share of would-be churners the offer actually saves
    horizon_years: float = 3.0       # how long a saved customer keeps generating value


def annual_customer_value(df: pd.DataFrame) -> np.ndarray:
    """Rough yearly revenue: ~2% margin on balances + fees per product + engagement bonus."""
    return 0.02 * df["balance"].to_numpy() + 70.0 * df["num_products"].to_numpy() \
        + 40.0 * df["is_active_member"].to_numpy()


def score_campaign(df: pd.DataFrame, p_churn: np.ndarray, a: CampaignAssumptions) -> pd.DataFrame:
    value = annual_customer_value(df)
    out = df[["customer_id"]].copy()
    out["p_churn"] = p_churn
    out["annual_value"] = value
    out["expected_gain"] = p_churn * a.offer_success_rate * value * a.horizon_years
    out["net_value"] = out["expected_gain"] - a.offer_cost
    out["contact"] = out["net_value"] > 0
    return out.sort_values("net_value", ascending=False).reset_index(drop=True)


def risk_tier(p: float) -> str:
    if p >= 0.60:
        return "Critical"
    if p >= 0.35:
        return "High"
    if p >= 0.15:
        return "Medium"
    return "Low"


def recommend_action(row: dict, drivers: list[str]) -> list[str]:
    """Rule-based playbook that maps *churn drivers* to *retention actions*."""
    acts = []
    if row["num_products"] >= 3:
        acts.append("Service review call: 3+ products often means friction or mis-selling; simplify the bundle.")
    if row["num_products"] == 1:
        acts.append("Cross-sell a second product (e.g. savings or card); 2-product customers churn the least.")
    if not row["is_active_member"]:
        acts.append("Re-engagement: app push, cashback on first card use, onboarding refresher.")
    if row["balance"] > 100_000 and not row["is_active_member"]:
        acts.append("Relationship manager outreach: large idle balance is a classic silent-leaver profile.")
    if row["geography"] == "Germany":
        acts.append("Counter local competition: market-specific rate or fee match.")
    if row["age"] >= 50:
        acts.append("Offer assisted channels (branch / phone) and retirement-oriented products.")
    if not acts:
        acts.append("No special action: keep in standard lifecycle communication.")
    return acts[:4]
