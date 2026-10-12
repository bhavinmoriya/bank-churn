# Bank Churn Lab

Learn churn modeling by building it from scratch: data -> model -> evaluation -> retention strategy -> Gradio app.

## Run
```bash
uv sync
uv run python -m bankchurn.train   # trains + writes artifacts/
uv run python app.py               # opens the Gradio app
uv run pytest                      # 4 tests (metrics vs sklearn, gradient check, ...)
```

## Learning path (read the files in this order)
| File | Concept |
|---|---|
| `bankchurn/data.py` | Synthetic customers with a *known* churn mechanism: check what the model rediscovers |
| `bankchurn/features.py` | Feature engineering for a linear model, scaling without leakage |
| `bankchurn/logreg.py` | Logistic regression + gradient descent, written in NumPy |
| `bankchurn/metrics.py` | Why accuracy fails; ROC-AUC, PR-AUC, recall, lift, calibration, from scratch |
| `bankchurn/strategy.py` | Turning probabilities into decisions: expected net value of a retention offer |
| `bankchurn/train.py` | Stratified split, baseline, scratch model vs gradient boosting |
| `app.py` | Gradio: single customer, campaign planner, model report |

## Key ideas
1. Churn is imbalanced (~15%): "nobody churns" is ~85% accurate and useless.
2. Rank customers (AUC, lift) and keep probabilities calibrated: they feed the money calculation.
3. Rank by **expected net value**, not churn probability: `P(churn) x P(offer works) x value x years - cost`.
4. Prediction is not causation: the model says who is *at risk*, not who the offer would *persuade* (uplift modeling is the next step).
