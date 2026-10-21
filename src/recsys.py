"""Top-N movie recommendation: data splitting, four recommenders, and ranking metrics.

Framing. A recommender's job is to put movies a user will like at the top of a
short list, so models are judged on *ranking* (did the held-out movies the user
rated 4+ stars show up in their top 10?) rather than on predicting star ratings.

All models share one interface:
    model.fit(X)                 X: users x items sparse matrix of training interactions
    model.score(X_users)         dense scores for those users over all items
and :func:`recommend` removes movies the user has already rated before taking
the top N.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds


# ---------------------------------------------------------------------------
# Splitting and matrices
# ---------------------------------------------------------------------------

def temporal_split(ratings: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user time split: each user's most recent ``test_frac`` of ratings
    become test data, everything before stays in train.

    This mimics the real task (predict what a user watches *next* from what
    they watched before) better than a random split, which would let the model
    train on a user's future.
    """
    r = ratings.sort_values(["userId", "timestamp", "movieId"]).copy()
    rank = r.groupby("userId").cumcount()
    size = r.groupby("userId")["movieId"].transform("size")
    n_test = np.maximum(1, np.floor(size * test_frac).astype(int))
    is_test = rank >= size - n_test
    return r[~is_test].reset_index(drop=True), r[is_test].reset_index(drop=True)


class Index:
    """Maps raw user and movie ids to matrix rows and columns."""

    def __init__(self, user_ids, item_ids):
        self.users = np.asarray(sorted(set(user_ids)))
        self.items = np.asarray(sorted(set(item_ids)))
        self.user_pos = {u: i for i, u in enumerate(self.users)}
        self.item_pos = {m: j for j, m in enumerate(self.items)}

    def matrix(self, ratings: pd.DataFrame, min_rating: float | None = None) -> sp.csr_matrix:
        """Binary users x items matrix; optionally keep only ratings >= min_rating."""
        r = ratings if min_rating is None else ratings[ratings["rating"] >= min_rating]
        r = r[r["movieId"].isin(self.item_pos) & r["userId"].isin(self.user_pos)]
        rows = r["userId"].map(self.user_pos).to_numpy()
        cols = r["movieId"].map(self.item_pos).to_numpy()
        data = np.ones(len(r), dtype=np.float64)
        X = sp.csr_matrix((data, (rows, cols)), shape=(len(self.users), len(self.items)))
        X.data[:] = 1.0  # collapse any duplicates to 1
        return X


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Popularity:
    """Recommend the movies with the most positive interactions in training.
    Same list for everyone: the baseline every personalised model must beat."""

    name = "Most popular"

    def fit(self, X):
        self.counts_ = np.asarray(X.sum(axis=0)).ravel()
        return self

    def score(self, X_users):
        return np.tile(self.counts_, (X_users.shape[0], 1))


class ItemKNN:
    """Item-based collaborative filtering ("people who liked X also liked Y").
    Cosine similarity between item columns, keep each item's ``k`` most similar
    neighbours, score = sum of similarities to the items the user liked."""

    name = "Item-kNN"

    def __init__(self, k: int = 100, shrink: float = 10.0):
        self.k, self.shrink = k, shrink

    def fit(self, X):
        X = sp.csc_matrix(X)
        co = (X.T @ X).toarray()
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + self.shrink + 1e-12)
        np.fill_diagonal(sim, 0.0)
        if self.k < sim.shape[0]:
            # keep the top-k neighbours in each column
            cutoff = -np.partition(-sim, self.k - 1, axis=0)[self.k - 1]
            sim[sim < cutoff] = 0.0
        self.sim_ = sim
        return self

    def score(self, X_users):
        return np.asarray(X_users @ self.sim_)


class PureSVD:
    """Matrix factorisation via truncated SVD of the interaction matrix.
    Each item becomes a ``factors``-dimensional vector; a user is scored by
    projecting their history onto those vectors: scores = X V V^T."""

    name = "PureSVD"

    def __init__(self, factors: int = 50, random_state: int = 0):
        self.factors, self.random_state = factors, random_state

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        v0 = rng.standard_normal(min(X.shape))
        _, _, vt = svds(sp.csr_matrix(X, dtype=np.float64), k=self.factors, v0=v0)
        self.V_ = vt.T
        return self

    def score(self, X_users):
        return np.asarray(X_users @ self.V_) @ self.V_.T


class EASE:
    """Embarrassingly Shallow Autoencoder (Steck, 2019).

    Learns an item-to-item weight matrix B by ridge regression that predicts
    each item from all the others, with the constraint that an item cannot
    predict itself. It has a closed-form solution:

        P = (X^T X + lambda I)^-1,   B = I - P / diag(P),   diag(B) = 0
    """

    name = "EASE"

    def __init__(self, lam: float = 200.0):
        self.lam = lam

    def fit(self, X):
        G = (sp.csc_matrix(X).T @ X).toarray()
        G[np.diag_indices_from(G)] += self.lam
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        np.fill_diagonal(B, 0.0)
        self.B_ = B
        return self

    def score(self, X_users):
        return np.asarray(X_users @ self.B_)


# ---------------------------------------------------------------------------
# Recommending and evaluating
# ---------------------------------------------------------------------------

def recommend(model, X_users, seen, n: int = 10) -> np.ndarray:
    """Top-``n`` item columns per user, never recommending an item in ``seen``."""
    scores = model.score(X_users).astype(np.float64)
    seen = sp.csr_matrix(seen)
    rows, cols = seen.nonzero()
    scores[rows, cols] = -np.inf
    top = np.argpartition(-scores, n, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def ranking_metrics(recs: np.ndarray, relevant: list[set], n_items: int, k: int = 10) -> dict:
    """Mean precision@k, recall@k, NDCG@k and hit rate@k over users that have at
    least one relevant held-out item, plus catalog coverage (share of all items
    recommended to at least one user)."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    prec, rec, ndcg, hit, per_user = [], [], [], [], []
    for row, rel in zip(recs, relevant):
        if not rel:
            per_user.append(np.nan)
            continue
        hits = np.array([item in rel for item in row[:k]], dtype=float)
        ideal = discounts[: min(len(rel), k)].sum()
        u_ndcg = float(hits @ discounts / ideal)
        prec.append(hits.sum() / k)
        rec.append(hits.sum() / len(rel))
        ndcg.append(u_ndcg)
        hit.append(float(hits.any()))
        per_user.append(u_ndcg)
    return {
        f"precision@{k}": float(np.mean(prec)),
        f"recall@{k}": float(np.mean(rec)),
        f"NDCG@{k}": float(np.mean(ndcg)),
        f"hit rate@{k}": float(np.mean(hit)),
        "coverage": len(np.unique(recs[:, :k])) / n_items,
        "users evaluated": len(ndcg),
        "_per_user_ndcg": np.array(per_user),
    }


def relevant_sets(index: Index, held_out: pd.DataFrame, min_rating: float = 4.0) -> list[set]:
    """For each user (in index order), the set of item columns they rated
    ``min_rating`` or higher in the held-out data. Movies that never appear in
    training have no column and can't be recommended by any model; they are
    left out of the relevant set but counted by :func:`unseen_share`."""
    pos = held_out[(held_out["rating"] >= min_rating) & held_out["movieId"].isin(index.item_pos)]
    by_user = pos.groupby("userId")["movieId"].apply(lambda s: {index.item_pos[m] for m in s})
    return [by_user.get(u, set()) for u in index.users]


def unseen_share(index: Index, held_out: pd.DataFrame, min_rating: float = 4.0) -> float:
    """Share of relevant held-out ratings whose movie never appears in training."""
    pos = held_out[held_out["rating"] >= min_rating]
    return float((~pos["movieId"].isin(index.item_pos)).mean())
