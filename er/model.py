"""GBDT matcher (LightGBM, MIT licence) + F0.5-aware decision rule."""
import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    seed=42,
)
NUM_ROUNDS = 600


def train_booster(X, y, rounds=NUM_ROUNDS, params=None):
    return lgb.train(params or LGB_PARAMS, lgb.Dataset(X, label=y), num_boost_round=rounds)


def oof_predict(X, y, groups, n_folds=5, rounds=NUM_ROUNDS):
    """Out-of-fold probabilities; folds are grouped by Source-1 entity so no entity leaks."""
    oof = np.zeros(len(y), np.float64)
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        b = train_booster(X.iloc[tr], y[tr], rounds)
        oof[va] = b.predict(X.iloc[va])
        print(f"  fold {k}: {len(va):,} pairs, pos={int(y[va].sum()):,}")
    return oof


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

def decide(li, ri, p, threshold, exclusive=True, rel=0.0):
    """Boolean mask of accepted pairs.

    exclusive: Source 1 is deduplicated, so an S2/S3 record can belong to at most
               one S1 entity -> keep only its highest-probability S1.
    rel:       additionally require p >= rel * (best p of that S1 entity).
    """
    p = np.asarray(p)
    keep = p >= threshold
    if exclusive:
        best_r = pd.Series(p).groupby(ri).transform("max").values
        keep &= p >= best_r
    if rel > 0:
        best_l = pd.Series(p).groupby(li).transform("max").values
        keep &= p >= rel * best_l
    return keep


def macro_f05_from_pairs(li, keep, label, n_true_per_left, beta=0.5):
    """Vectorised macro F0.5. n_true_per_left: array (len = #S1) of true match counts
    (counting positives that blocking missed, so recall is honest)."""
    n = len(n_true_per_left)
    tp = np.bincount(li, weights=(keep & (label == 1)).astype(float), minlength=n)
    npred = np.bincount(li, weights=keep.astype(float), minlength=n)
    ntrue = n_true_per_left.astype(float)
    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / npred, 0)
        rec = np.where(ntrue > 0, tp / ntrue, 0)
        f = np.where(tp > 0, (1 + b2) * prec * rec / (b2 * prec + rec), 0.0)
    f = np.where((ntrue == 0) & (npred == 0), 1.0, f)
    return float(f.mean())


def tune_decision(li, ri, p, label, n_true_per_left):
    """Grid-search the decision rule directly on macro F0.5 (the leaderboard metric)."""
    p = np.asarray(p)
    is_best_r = p >= pd.Series(p).groupby(ri).transform("max").values
    best_l = pd.Series(p).groupby(li).transform("max").values
    best = (-1, None)
    grid = itertools.product(np.round(np.arange(0.05, 0.96, 0.025), 3), [True, False], [0.0, 0.3, 0.6, 0.8])
    for t, ex, rel in grid:
        keep = (p >= t) & (is_best_r if ex else True) & (p >= rel * best_l)
        s = macro_f05_from_pairs(li, keep, label, n_true_per_left)
        if s > best[0]:
            best = (s, dict(threshold=float(t), exclusive=bool(ex), rel=float(rel)))
    return best
