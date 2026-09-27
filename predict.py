"""Produce output/matching_results.tsv and output/candidate_pairs.tsv for the test set.

    python predict.py --data dataset --artifacts artifacts --out output
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np

from er.blocking import BlockingConfig
from er.data import load_split
from er.model import decide
from er.pipeline import block_and_featurize, to_sets, write_id_lists
from validate_submission import validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    with open(os.path.join(args.artifacts, "config.json")) as f:
        conf = json.load(f)
    cfg = BlockingConfig(**conf["blocking"])
    booster = lgb.Booster(model_file=os.path.join(args.artifacts, "model.txt"))

    left, right = load_split(args.data, args.split)
    cand, X = block_and_featurize(left, right, cfg)
    X = X[conf["features"]]
    li, ri = cand["li"].values, cand["ri"].values
    p = booster.predict(X)
    keep = decide(li, ri, p, **conf["decision"])

    order = left["entity_id"].tolist()
    os.makedirs(args.out, exist_ok=True)
    cand_path = os.path.join(args.out, "candidate_pairs.tsv")
    match_path = os.path.join(args.out, "matching_results.tsv")
    write_id_lists(to_sets(left, right, li, ri), order, "candidate_entity_ids", cand_path)
    write_id_lists(to_sets(left, right, li, ri, keep), order, "matched_entity_ids", match_path)
    n_match = int(keep.sum())
    print(f"wrote {match_path}: {n_match:,} links, "
          f"{np.mean([len(s) == 0 for s in to_sets(left, right, li, ri, keep).values()]):.1%} empty rows")
    print(f"wrote {cand_path}: {len(cand):,} pairs ({len(cand) / max(len(left), 1):.2f} per S1)")
    validate(args.data, args.split, match_path, cand_path)


if __name__ == "__main__":
    main()
