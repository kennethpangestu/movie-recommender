import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recsys import (  # noqa: E402
    EASE,
    Index,
    ItemKNN,
    Popularity,
    PureSVD,
    ranking_metrics,
    recommend,
    relevant_sets,
    temporal_split,
)


def _ratings():
    rows = []
    for user in range(1, 4):
        for t, movie in enumerate(range(10, 20)):
            rows.append({"userId": user, "movieId": movie + user, "rating": 4.0 if t % 2 else 3.0,
                         "timestamp": 1000 * user + t})
    return pd.DataFrame(rows)


def test_temporal_split_holds_out_each_users_latest_ratings():
    r = _ratings()
    train, test = temporal_split(r, test_frac=0.2)
    assert len(test) == 3 * 2 and len(train) == 3 * 8
    for user in r["userId"].unique():
        assert train.loc[train["userId"] == user, "timestamp"].max() < \
            test.loc[test["userId"] == user, "timestamp"].min()


def test_recommend_never_returns_seen_items():
    X = sp.csr_matrix(np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [1, 0, 1, 1, 0]], dtype=float))
    recs = recommend(Popularity().fit(X), X, X, n=2)
    for u in range(3):
        assert not set(recs[u]) & set(X[u].indices)


def test_models_score_co_liked_item_highest():
    # Items 0 and 1 are always liked together; item 2 is liked by different users.
    X = sp.csr_matrix(np.array([
        [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0],
    ], dtype=float))
    user = sp.csr_matrix(np.array([[1, 0, 0, 0]], dtype=float))  # only liked item 0
    for model in [ItemKNN(k=3, shrink=0), EASE(lam=1.0), PureSVD(factors=2)]:
        scores = model.fit(X).score(user).ravel()
        assert np.argmax(np.delete(scores, 0)) + 1 == 1, model.name


def test_ease_has_zero_diagonal():
    X = sp.random(30, 8, density=0.4, random_state=0, format="csr")
    X.data[:] = 1
    B = EASE(lam=5).fit(X).B_
    assert np.allclose(np.diag(B), 0)


def test_ranking_metrics_hand_computed():
    recs = np.array([[0, 1, 2], [3, 4, 5]])
    relevant = [{0, 2}, set()]  # second user has nothing relevant and is skipped
    m = ranking_metrics(recs, relevant, n_items=10, k=3)
    assert m["users evaluated"] == 1
    assert m["precision@3"] == pytest.approx(2 / 3)
    assert m["recall@3"] == pytest.approx(1.0)
    ideal = 1 + 1 / np.log2(3)
    assert m["NDCG@3"] == pytest.approx((1 + 1 / np.log2(4)) / ideal)
    assert m["coverage"] == pytest.approx(0.6)


def test_relevant_sets_only_counts_high_ratings_on_known_items():
    idx = Index([1, 2], [10, 11])
    held = pd.DataFrame({"userId": [1, 1, 1, 2], "movieId": [10, 11, 99, 11],
                         "rating": [5.0, 3.0, 5.0, 4.0], "timestamp": [1, 2, 3, 4]})
    assert relevant_sets(idx, held) == [{0}, {1}]
