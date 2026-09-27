"""Official metric: F_0.5 computed per Source-1 entity, then macro-averaged (singletons included)."""
import numpy as np


def f05_single(pred: set, true: set, beta: float = 0.5) -> float:
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f05(pred: dict, truth: dict, ids) -> float:
    return float(np.mean([f05_single(pred.get(i, set()), truth.get(i, set())) for i in ids]))


def blocking_report(cand_sets: dict, truth: dict, ids, n_right: int) -> dict:
    """Recall ceiling / pairs-per-entity / reduction ratio / oracle F0.5 of a candidate set."""
    tot_true = sum(len(truth.get(i, ())) for i in ids)
    hit = sum(len(truth.get(i, set()) & cand_sets.get(i, set())) for i in ids)
    n_pairs = sum(len(cand_sets.get(i, ())) for i in ids)
    oracle = {i: truth.get(i, set()) & cand_sets.get(i, set()) for i in ids}
    return {
        "pair_recall": hit / max(tot_true, 1),
        "avg_candidates_per_s1": n_pairs / max(len(ids), 1),
        "max_candidates_per_s1": max((len(cand_sets.get(i, ())) for i in ids), default=0),
        "reduction_ratio": 1 - n_pairs / max(len(ids) * n_right, 1),
        "oracle_macro_f05": macro_f05(oracle, truth, ids),
    }
