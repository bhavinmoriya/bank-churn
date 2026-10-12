import numpy as np

from bankchurn import metrics as M
from bankchurn.data import generate_customers
from bankchurn.features import Standardizer, engineer
from bankchurn.logreg import LogisticRegressionScratch


def test_roc_auc_matches_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    assert abs(M.roc_auc(y, p) - roc_auc_score(y, p)) < 1e-9
    assert abs(M.average_precision(y, p) - average_precision_score(y, p)) < 1e-9


def test_gradient_is_correct():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(200, 3)), rng.integers(0, 2, 200).astype(float)
    m = LogisticRegressionScratch(n_iter=1).fit(X, y)  # sets shapes
    w, b = np.array([0.3, -0.2, 0.1]), 0.05

    def loss(w, b):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    p = 1 / (1 + np.exp(-(X @ w + b)))
    gw = X.T @ (p - y) / len(y)
    num = np.array([(loss(w + 1e-6 * e, b) - loss(w - 1e-6 * e, b)) / 2e-6 for e in np.eye(3)])
    assert np.allclose(gw, num, atol=1e-6)


def test_model_beats_baseline_and_loss_decreases():
    df = generate_customers(6000, 3)
    tr, te = M.train_test_split_stratified(df.churned.to_numpy(), 0.25, 3)
    X = engineer(df).to_numpy()
    sc = Standardizer().fit(X[tr])
    m = LogisticRegressionScratch().fit(sc.transform(X[tr]), df.churned.to_numpy()[tr])
    assert m.loss_history_[-1] < m.loss_history_[0]
    assert M.roc_auc(df.churned.to_numpy()[te], m.predict_proba(sc.transform(X[te]))) > 0.75


def test_split_is_stratified_and_disjoint():
    y = (np.random.default_rng(0).random(1000) < 0.2).astype(int)
    tr, te = M.train_test_split_stratified(y, 0.25, 0)
    assert not set(tr) & set(te)
    assert abs(y[tr].mean() - y[te].mean()) < 0.01
