I built the project: a churn model written from scratch, a retention strategy layer, and a three-tab Gradio app. I ran the training, the 4 tests, and the app's functions, but I haven't opened the app in a browser. It's in `bank-churn.zip`.

To run it:
```bash
uv sync
uv run python -m bankchurn.train
uv run python app.py
```

**What's in it**
- **Data:** The data is synthetic, because real bank churn data is private. I wrote the churn mechanism myself, so you can check whether the model finds the reasons I built in. For example, customers with 3 or more products, inactive members, and customers in Germany churn more.
- **Model:** Logistic regression is written in plain NumPy with gradient descent. For comparison I also trained a gradient boosting model from scikit-learn.
- **Metrics:** ROC-AUC, PR-AUC, recall, lift and calibration are all written by hand. Two of them are checked against scikit-learn in the tests.
- **App:** The three tabs are:
  1. **Single customer:** Enter a customer's details to get a stay-or-leave prediction, the main reasons, and suggested retention actions.
  2. **Retention campaign:** Set the offer cost and success rate to get a ranked list of customers to contact and where to stop spending.
  3. **Model report:** Shows how well the model performs.

**Results**

| | Logistic regression | Gradient boosting |
|---|---|---|
| ROC-AUC | 0.810 | 0.794 |
| Top-decile lift | 3.8x | 3.6x |

About 15% of customers churn. A model that predicts "nobody leaves" would be 85% accurate and useless, which is why accuracy isn't used. The riskiest 10% of customers churn at about 3.8 times the average rate.

The simple model beats gradient boosting here because I generated the data from a logistic formula, so it fits perfectly. On real bank data, gradient boosting usually wins, and that is worth testing.

**My take on the project**
1. **Rank by expected value, not churn probability.** A good retention strategy doesn't call everyone with a high score. It ranks customers by P(churn) × P(offer works) × customer value × years − offer cost. A high-value customer at moderate risk can beat a low-value customer at high risk.
2. **Next step: uplift modeling.** The model says who is at risk, not who an offer would actually persuade. Some at-risk customers leave anyway, and some would have stayed without the offer. Uplift modeling needs A/B test data and answers that question. I'd add it once the basics make sense.
3. **Use real data when you can.** Swap in a public bank churn dataset such as the Kaggle one. It uses the same columns, so only `data.py` has to change.

To learn it, read the files in the order listed in `README.md`. Each one covers one concept.
