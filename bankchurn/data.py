"""Synthetic retail-banking customers with a *known* churn mechanism.

Why synthetic? Real churn data is private. Here we write down the "true" reasons
customers leave, so you can check whether the model rediscovers them. The
patterns mirror what is commonly seen in public bank-churn datasets:

* customers with 1 product are at risk, 2 products are the sweet spot,
  3-4 products signal a bad experience (mis-selling / friction)
* inactive members leave far more often
* middle-aged to older customers leave more than young ones
* one market (Germany) has stronger competition -> higher churn
* high balances with low engagement are a classic "silent leaver" profile
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "credit_score", "geography", "gender", "age", "tenure", "balance",
    "num_products", "has_credit_card", "is_active_member", "estimated_salary",
]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate_customers(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    geography = rng.choice(["France", "Germany", "Spain"], size=n, p=[0.50, 0.25, 0.25])
    gender = rng.choice(["Female", "Male"], size=n, p=[0.45, 0.55])
    age = np.clip(rng.normal(39, 10.5, n), 18, 85).round().astype(int)
    tenure = rng.integers(0, 11, n)
    credit_score = np.clip(rng.normal(650, 95, n), 350, 850).round().astype(int)
    is_active = rng.binomial(1, 0.52, n)
    has_card = rng.binomial(1, 0.70, n)
    salary = np.clip(rng.lognormal(11.3, 0.55, n), 12_000, 250_000).round(2)

    # ~35% have zero balance; the rest are roughly normal. Germany: richer balances.
    has_bal = rng.random(n) > 0.35
    balance = np.where(has_bal, np.clip(rng.normal(95_000, 35_000, n), 1_000, 250_000), 0.0)
    balance = np.where((geography == "Germany") & has_bal,
                       balance * 1.25, balance).round(2)

    num_products = np.where(
        rng.random(n) < 0.06 + 0.10 * (balance > 0) * (is_active == 0), 3,
        np.where(rng.random(n) < 0.50, 1, 2),
    )
    num_products = np.where(rng.random(n) < 0.012, 4, num_products)

    # --- the hidden truth -------------------------------------------------
    logit = -2.9
    logit += 0.045 * (age - 40) + 0.0009 * np.maximum(age - 45, 0) ** 2
    logit += np.where(geography == "Germany", 0.85, 0.0)
    logit += np.where(gender == "Female", 0.28, 0.0)
    logit += np.where(is_active == 0, 0.95, 0.0)
    logit += np.select([num_products == 1, num_products == 2, num_products == 3, num_products == 4],
                       [0.35, -0.95, 1.9, 3.5])
    logit += 0.35 * (balance > 100_000)
    logit += -0.0012 * (credit_score - 650) / 10
    logit += -0.04 * tenure * 0.5
    logit += 0.15 * rng.normal(size=n)  # noise nobody can explain

    churn = rng.binomial(1, _sigmoid(logit))

    df = pd.DataFrame({
        "credit_score": credit_score, "geography": geography, "gender": gender,
        "age": age, "tenure": tenure, "balance": balance,
        "num_products": num_products, "has_credit_card": has_card,
        "is_active_member": is_active, "estimated_salary": salary,
        "churned": churn,
    })
    df.insert(0, "customer_id", np.arange(100_000, 100_000 + n))
    return df


if __name__ == "__main__":
    d = generate_customers()
    print(d.head())
    print("churn rate:", d.churned.mean().round(3))
