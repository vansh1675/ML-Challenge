"""Train the matcher on dataset/train and tune the decision rule with out-of-fold F0.5.

    python train.py --data dataset --artifacts artifacts
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from er.blocking import BlockingConfig
from er.data import load_ground_truth, load_split
from er.metrics import blocking_report, macro_f05
from er.model import decide, oof_predict, train_booster, tune_decision, NUM_ROUNDS
from er.pipeline import block_and_featurize, to_sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    for k, v in BlockingConfig().to_dict().items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()
    cfg = BlockingConfig(**{k: getattr(args, k) for k in BlockingConfig().to_dict()})
    os.makedirs(args.artifacts, exist_ok=True)

    left, right = load_split(args.data, "train")
    truth = load_ground_truth(os.path.join(args.data, "train", "train_ground_truth.tsv"))
    ids = left["entity_id"].tolist()
    n_true = np.array([len(truth.get(i, ())) for i in ids])
    print(f"train: {len(left):,} S1, {len(right):,} S2/S3, {n_true.sum():,} true links, "
          f"{(n_true == 0).mean():.1%} singletons")

    cand, X = block_and_featurize(left, right, cfg)
    li, ri = cand["li"].values, cand["ri"].values
    rid = right["entity_id"].values
    y = np.array([rid[r] in truth.get(ids[l], ()) for l, r in zip(li, ri)], np.int8)

    rep = blocking_report(to_sets(left, right, li, ri), truth, ids, len(right))
    print("blocking report:", json.dumps({k: round(v, 4) for k, v in rep.items()}))

    print("out-of-fold training ...")
    oof = oof_predict(X, y, groups=li, n_folds=args.folds, rounds=args.rounds)
    score, dec = tune_decision(li, ri, oof, y, n_true)
    print(f"OOF macro F0.5 = {score:.4f} with {dec}")

    # sanity: recompute with the reference (set-based) metric
    keep = decide(li, ri, oof, **dec)
    ref = macro_f05(to_sets(left, right, li, ri, keep), truth, ids)
    print(f"OOF macro F0.5 (reference implementation) = {ref:.4f}")
    for c in sorted(set(left["country_n"])):
        sub = [i for i, cc in zip(ids, left["country_n"]) if cc == c]
        print(f"  country={c or '<empty>'}: {macro_f05(to_sets(left, right, li, ri, keep), truth, sub):.4f} "
              f"({len(sub):,} S1)")

    print("fitting final model on all training pairs ...")
    booster = train_booster(X, y, args.rounds)
    booster.save_model(os.path.join(args.artifacts, "model.txt"))
    imp = pd.Series(booster.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print("top features:\n" + imp.head(15).round(0).to_string())

    with open(os.path.join(args.artifacts, "config.json"), "w") as f:
        json.dump({"blocking": cfg.to_dict(), "decision": dec, "features": list(X.columns),
                   "oof_macro_f05": score, "blocking_report": rep}, f, indent=2)
    print(f"saved to {args.artifacts}/")


if __name__ == "__main__":
    main()
