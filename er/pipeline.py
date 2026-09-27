"""Shared steps used by train.py and predict.py."""
import time

import numpy as np
import pandas as pd

from .blocking import BlockingConfig, Vectors, generate_candidates
from .features import build_features


def block_and_featurize(left, right, cfg: BlockingConfig):
    t0 = time.time()
    vecs = Vectors(left, right, cfg)
    cand = generate_candidates(left, right, vecs, cfg)
    t1 = time.time()
    print(f"blocking: {len(cand):,} candidate pairs for {len(left):,} S1 x {len(right):,} S2/S3 "
          f"({len(cand) / max(len(left), 1):.2f} per S1) in {t1 - t0:.1f}s")
    X = build_features(cand, left, right, vecs)
    print(f"features: {X.shape[1]} features in {time.time() - t1:.1f}s")
    return cand, X


def to_sets(left, right, li, ri, mask=None):
    """Positional pairs -> {s1_id: set(right ids)} covering every S1 entity."""
    li, ri = np.asarray(li), np.asarray(ri)
    if mask is not None:
        li, ri = li[mask], ri[mask]
    out = {i: set() for i in left["entity_id"]}
    lid = left["entity_id"].values[li]
    rid = right["entity_id"].values[ri]
    for a, b in zip(lid, rid):
        out[a].add(b)
    return out


def write_id_lists(sets: dict, order, col: str, path: str):
    rows = [(i, ",".join(sorted(sets.get(i, set())))) for i in order]
    pd.DataFrame(rows, columns=["source1_entity_id", col]).to_csv(path, sep="\t", index=False)
