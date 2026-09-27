"""Sweep blocking settings on the training set: recall ceiling vs. candidates per S1.

The final ranking rewards *small* candidate sets, so pick the cheapest config whose
pair recall / oracle F0.5 is still close to the widest one, then pass its values to
train.py (e.g. --k_left 3 --k_right 1 ...).

    python tune_blocking.py --data dataset --min_recall 0.98
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

from er.blocking import BlockingConfig, Vectors, add_ranks, candidate_pool, select
from er.data import load_ground_truth, load_split
from er.model import macro_f05_from_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--min_recall", type=float, default=0.98)
    ap.add_argument("--name_weight", type=float, default=BlockingConfig.name_weight)
    ap.add_argument("--sample_s1", type=int, default=100_000,
                    help="evaluate selection on this many random S1 entities (ranks use the full pool)")
    a = ap.parse_args()

    left, right = load_split(a.data, "train")
    truth = load_ground_truth(os.path.join(a.data, "train", "train_ground_truth.tsv"))
    ids = left["entity_id"].tolist()
    base = BlockingConfig(name_weight=a.name_weight)
    vecs = Vectors(left, right, base)
    raw = add_ranks(candidate_pool(left, right, vecs, base))  # wide pool once; selection re-run per config
    rid = right["entity_id"].values
    raw["y"] = [rid[r] in truth.get(ids[l], ()) for l, r in zip(raw["li"], raw["ri"])]
    n_true = np.array([len(truth.get(i, ())) for i in ids])
    if a.sample_s1 and a.sample_s1 < len(ids):
        chosen = np.zeros(len(ids), bool)
        chosen[np.random.default_rng(0).choice(len(ids), a.sample_s1, replace=False)] = True
        remap = -np.ones(len(ids), np.int64)
        remap[chosen] = np.arange(chosen.sum())
        raw = raw[chosen[raw["li"].values]].copy()
        raw["li"] = remap[raw["li"].values]
        n_true = n_true[chosen]
        ids = [i for i, c in zip(ids, chosen) if c]

    rows = []
    y_pool = raw["y"].values.astype(bool)
    print(f"pool: {len(raw) / len(ids):.2f} pairs per S1, pool recall = {y_pool.sum() / max(n_true.sum(), 1):.4f}")
    grid = itertools.product([1, 2, 3], [1, 2], [0, 1], [0, 1], [0.0, 0.3], [0.0, 0.7], [3, 5, 8])
    for kl, kr, kn, ke, ms, rel, cap in grid:
        cfg = BlockingConfig(k_left=kl, k_right=kr, k_name=kn, keep_exact_name=ke, min_sim=ms,
                             rel_sim=rel, max_per_s1=cap, name_weight=a.name_weight)
        c = select(raw, cfg)
        y = c["y"].values.astype(bool)
        sel_keys = ["k_left", "k_right", "k_name", "keep_exact_name", "min_sim", "min_name_sim",
                    "rel_sim", "max_per_s1"]
        rows.append({**{k: getattr(cfg, k) for k in sel_keys},
                     "pair_recall": y.sum() / max(n_true.sum(), 1),
                     "avg_candidates_per_s1": len(c) / len(ids),
                     # best achievable score if the matcher were perfect on these candidates
                     "oracle_macro_f05": macro_f05_from_pairs(c["li"].values, y, y.astype(int), n_true)})
    df = pd.DataFrame(rows).drop_duplicates(["pair_recall", "avg_candidates_per_s1"])
    df = df.sort_values("avg_candidates_per_s1")
    print(f"widest recall = {df['pair_recall'].max():.4f}")
    ok = df[df["pair_recall"] >= a.min_recall]
    pd.set_option("display.width", 200)
    print((ok if len(ok) else df.sort_values("pair_recall", ascending=False)).head(20).round(4).to_string(index=False))
    df.to_csv("blocking_sweep.tsv", sep="\t", index=False)
    print("full sweep -> blocking_sweep.tsv")


if __name__ == "__main__":
    main()
