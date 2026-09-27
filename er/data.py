"""Loading + per-record preprocessing."""
import os

import pandas as pd

from .text import acronym, norm_address, norm_country, norm_name

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path):
    # keep_default_na=False: an empty field must stay "" (and business names like "NA" must survive)
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[], quoting=3)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    df = df.fillna("")
    names = df["business_name"].map(norm_name)
    df["name_full"] = names.map(lambda x: x[0])
    df["name_core"] = names.map(lambda x: x[1])
    df["name_alias"] = names.map(lambda x: x[2])
    df["name_acr"] = df["name_core"].map(acronym)
    addrs = df["business_address"].map(norm_address)
    df["addr"] = addrs.map(lambda x: x[0])
    df["addr_nums"] = addrs.map(lambda x: x[1])
    df["postal"] = addrs.map(lambda x: x[2])
    df["country_n"] = df["country"].map(norm_country)
    df["source"] = df["entity_id"].str.slice(0, 2)
    return df.reset_index(drop=True)


def load_split(data_dir: str, split: str):
    """split in {'train','test'}; returns (s1, right) where right = source2 + source3 stacked."""
    d = os.path.join(data_dir, split)
    s1 = read_tsv(os.path.join(d, f"{split}_source1.tsv"))
    s2 = read_tsv(os.path.join(d, f"{split}_source2.tsv"))
    s3 = read_tsv(os.path.join(d, f"{split}_source3.tsv"))
    right = pd.concat([s2, s3], ignore_index=True)
    return prepare(s1), prepare(right)


def load_ground_truth(path: str) -> dict:
    gt = read_tsv(path)
    out = {}
    for s1_id, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        out[s1_id] = set(x.strip() for x in str(ids).split(",") if x.strip())
    return out
