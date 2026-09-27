"""Checks both output files against every rule in the challenge statement.

    python validate_submission.py --data dataset --split test --out output
"""
import argparse
import os

from er.data import read_tsv


def _parse(path, col):
    df = read_tsv(path)
    assert list(df.columns) == ["source1_entity_id", col], f"{path}: bad header {list(df.columns)}"
    return {a: [x for x in b.split(",") if x] for a, b in zip(df["source1_entity_id"], df[col])}, df


def validate(data, split, match_path, cand_path):
    d = os.path.join(data, split)
    s1 = set(read_tsv(os.path.join(d, f"{split}_source1.tsv"))["entity_id"])
    other = set(read_tsv(os.path.join(d, f"{split}_source2.tsv"))["entity_id"]) | \
        set(read_tsv(os.path.join(d, f"{split}_source3.tsv"))["entity_id"])
    lists = {}
    for path, col in [(match_path, "matched_entity_ids"), (cand_path, "candidate_entity_ids")]:
        m, df = _parse(path, col)
        assert df["source1_entity_id"].is_unique, f"{path}: duplicate source1 rows"
        assert set(m) == s1, f"{path}: missing {len(s1 - set(m))} / extra {len(set(m) - s1)} S1 ids"
        for k, v in m.items():
            assert len(v) == len(set(v)), f"{path}: duplicate ids in list of {k}"
            bad = [x for x in v if x not in other]
            assert not bad, f"{path}: {k} references unknown / non S2-S3 ids {bad[:3]}"
        lists[col] = m
    for k, v in lists["matched_entity_ids"].items():
        miss = set(v) - set(lists["candidate_entity_ids"][k])
        assert not miss, f"{k}: matched ids not in candidate_pairs {miss}"
    print("validation OK: both files are well-formed and matches are a subset of candidates")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    validate(a.data, a.split, os.path.join(a.out, "matching_results.tsv"),
             os.path.join(a.out, "candidate_pairs.tsv"))
