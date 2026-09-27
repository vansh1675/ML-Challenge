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

from er.blocking import BlockingConfig, Vectors, prune, raw_neighbours
from er.data import load_ground_truth, load_split
from er.model import macro_f05_from_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--min_recall", type=float, default=0.98)
    ap.add_argument("--name_weight", type=float, default=BlockingConfig.name_weight)
    a = ap.parse_args()

    left, right = load_split(a.data, "train")
    truth = load_ground_truth(os.path.join(a.data, "train", "train_ground_truth.tsv"))
    ids = left["entity_id"].tolist()
    base = BlockingConfig(name_weight=a.name_weight)
    vecs = Vectors(left, right, base)
    raw = raw_neighbours(left, right, vecs, 10, 3, 5, base.chunk_cells)
    rid = right["entity_id"].values
    raw["y"] = [rid[r] in truth.get(ids[l], ()) for l, r in zip(raw["li"], raw["ri"])]
    n_true = np.array([len(truth.get(i, ())) for i in ids])

    rows = []
    grid = itertools.product([1, 2, 3, 5, 10], [0, 1, 2, 3], [0, 1, 2, 5],
                             [0.0, 0.3, 0.4], [0.0, 0.6, 0.8], [3, 5, 8])
    for kl, kr, kn, ms, rel, cap in grid:
        cfg = BlockingConfig(k_left=kl, k_right=kr, k_name=kn, min_sim=ms, min_name_sim=0.55,
                             rel_sim=rel, max_per_s1=cap, name_weight=a.name_weight)
        c = prune(raw, cfg)
        y = c["y"].values.astype(bool)
        rows.append({**{k: v for k, v in cfg.to_dict().items() if k != "chunk_cells"},
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
