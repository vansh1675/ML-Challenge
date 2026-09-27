"""Scalable candidate generation (blocking).

Idea: every record is embedded as a sparse, L2-normalised TF-IDF vector
(char n-grams of the cleaned name + word tokens of the cleaned address). Inside
each country block we retrieve nearest neighbours in *both* directions:

  * S1 -> S2/S3 : top-k_left neighbours of every Source-1 entity
  * S2/S3 -> S1 : top-k_right neighbours of every Source-2/3 record. Source 1 is
                  deduplicated, so a S2/S3 record belongs to at most one S1
                  entity; asking "which S1 is closest to me?" recovers matches
                  for S1 entities that have many duplicates.
  * S1 -> S2/S3 : top-k_name neighbours on the name vector alone (handles
                  records with a missing / landmark-style address).

The union is then pruned with absolute and relative similarity cut-offs, so
candidates per S1 stay tiny. Cost is O(N * k) memory; the dense block product
is computed in row chunks. For billions of records the exact same vectors can be
dropped into an ANN index (FAISS / HNSW) per country - nothing else changes.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass
class BlockingConfig:
    k_left: int = 5          # S1 -> R neighbours on the combined vector
    k_right: int = 2         # R -> S1 neighbours on the combined vector
    k_name: int = 2          # S1 -> R neighbours on the name vector only
    name_weight: float = 0.8  # weight of the name block in the combined vector
    min_sim: float = 0.30    # absolute floor on combined cosine ...
    min_name_sim: float = 0.55  # ... unless the name alone is this similar
    rel_sim: float = 0.60    # keep only candidates within rel_sim * best (per S1 and per R)
    max_per_s1: int = 8      # hard cap per Source-1 entity
    chunk_cells: int = 30_000_000  # dense cells per matmul chunk (memory knob)

    def to_dict(self):
        return asdict(self)


class Vectors:
    """Fits TF-IDF on left+right texts and exposes row-normalised matrices."""

    def __init__(self, left: pd.DataFrame, right: pd.DataFrame, cfg: BlockingConfig):
        nl = len(left)
        names = pd.concat([left["name_core"], right["name_core"]], ignore_index=True)
        # postal code is repeated so that it carries extra weight inside the address block
        addrs = pd.concat([left["addr"] + " " + left["postal"], right["addr"] + " " + right["postal"]],
                          ignore_index=True)

        self.name_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True,
                                         min_df=1, dtype=np.float32)
        self.name_word = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True,
                                         dtype=np.float32)
        self.addr_word = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True,
                                         dtype=np.float32)
        self.addr_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True,
                                         dtype=np.float32)
        Nc = self.name_char.fit_transform(names.where(names != "", "_"))
        Nw = self.name_word.fit_transform(names.where(names != "", "_"))
        Aw = self.addr_word.fit_transform(addrs.where(addrs.str.strip() != "", "_"))
        Ac = self.addr_char.fit_transform(addrs.where(addrs.str.strip() != "", "_"))

        w = cfg.name_weight
        comb = normalize(sp.hstack([w * Nc, np.sqrt(1 - w * w) * Aw]).tocsr())
        self.mats = {"name_char": Nc.tocsr(), "name_word": Nw.tocsr(), "addr_word": Aw.tocsr(),
                     "addr_char": Ac.tocsr(), "comb": comb}
        self.nl = nl

    def left(self, key):
        return self.mats[key][: self.nl]

    def right(self, key):
        return self.mats[key][self.nl:]

    def pair_cosine(self, key, li, ri):
        """Row-wise cosine for aligned index arrays (vectors are already L2-normalised)."""
        L = self.left(key)[li]
        R = self.right(key)[ri]
        return np.asarray(L.multiply(R).sum(axis=1)).ravel().astype(np.float32)


def _topk(L, R, k, chunk_cells):
    """Exact top-k cosine neighbours of each row of L among rows of R -> (row, col, rank)."""
    n, m = L.shape[0], R.shape[0]
    if n == 0 or m == 0 or k <= 0:
        e = np.empty(0, np.int64)
        return e, e, e
    k = min(k, m)
    RT = R.T.tocsr()
    step = max(1, chunk_cells // m)
    out_l, out_r = [], []
    for s in range(0, n, step):
        S = (L[s:s + step] @ RT).toarray()
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1, kind="stable")
        idx = np.take_along_axis(idx, order, 1)
        out_l.append(np.repeat(np.arange(s, s + S.shape[0]), k))
        out_r.append(idx.ravel())
    rows, cols = np.concatenate(out_l), np.concatenate(out_r)
    return rows, cols, np.tile(np.arange(k), len(rows) // k)


def _country_groups(left, right):
    """Blocks by country label (open set). Records with an empty country join every block."""
    lc, rc = left["country_n"].values, right["country_n"].values
    countries = sorted((set(lc) | set(rc)) - {""})
    if not countries:
        yield np.arange(len(left)), np.arange(len(right))
        return
    for c in countries:
        yield np.where((lc == c) | (lc == ""))[0], np.where((rc == c) | (rc == ""))[0]


BIG = 10 ** 6


def raw_neighbours(left, right, vecs: Vectors, k_left, k_right, k_name, chunk_cells) -> pd.DataFrame:
    """Union of the three kNN lists with the rank each pair got in each list (BIG = absent)."""
    Lc, Rc = vecs.left("comb"), vecs.right("comb")
    Ln, Rn = vecs.left("name_char"), vecs.right("name_char")
    parts = []
    for gl, gr in _country_groups(left, right):
        if len(gl) == 0 or len(gr) == 0:
            continue
        a, b, rk = _topk(Lc[gl], Rc[gr], k_left, chunk_cells)
        parts.append(pd.DataFrame({"li": gl[a], "ri": gr[b], "rk_left": rk}))
        b, a, rk = _topk(Rc[gr], Lc[gl], k_right, chunk_cells)
        parts.append(pd.DataFrame({"li": gl[a], "ri": gr[b], "rk_right": rk}))
        a, b, rk = _topk(Ln[gl], Rn[gr], k_name, chunk_cells)
        parts.append(pd.DataFrame({"li": gl[a], "ri": gr[b], "rk_name": rk}))
    cols = ["li", "ri", "rk_left", "rk_right", "rk_name"]
    if not parts:
        return pd.DataFrame(columns=cols + ["sim_comb", "sim_name"])
    raw = pd.concat(parts, ignore_index=True).reindex(columns=cols).fillna(BIG)
    raw = raw.groupby(["li", "ri"], as_index=False).min().astype(np.int64)
    raw["sim_comb"] = vecs.pair_cosine("comb", raw["li"].values, raw["ri"].values)
    raw["sim_name"] = vecs.pair_cosine("name_char", raw["li"].values, raw["ri"].values)
    return raw


def generate_candidates(left, right, vecs: Vectors, cfg: BlockingConfig) -> pd.DataFrame:
    """Returns DataFrame[li, ri, sim_comb, sim_name] of candidate pairs (positional indices)."""
    raw = raw_neighbours(left, right, vecs, cfg.k_left, cfg.k_right, cfg.k_name, cfg.chunk_cells)
    return prune(raw, cfg)


def prune(cand: pd.DataFrame, cfg: BlockingConfig) -> pd.DataFrame:
    """Applies rank limits (lets one wide kNN pass be re-used for many configs) + similarity cut-offs."""
    keep = (cand["rk_left"] < cfg.k_left) | (cand["rk_right"] < cfg.k_right) | (cand["rk_name"] < cfg.k_name)
    keep &= (cand["sim_comb"] >= cfg.min_sim) | (cand["sim_name"] >= cfg.min_name_sim)
    cand = cand[keep]
    score = np.maximum(cand["sim_comb"], cand["sim_name"])
    best_l = score.groupby(cand["li"]).transform("max")
    best_r = score.groupby(cand["ri"]).transform("max")
    # a pair survives if it is competitive from the S1 side OR from the R side
    keep = (score >= cfg.rel_sim * best_l) | (score >= cfg.rel_sim * best_r)
    cand = cand[keep].copy()
    cand["_s"] = np.maximum(cand["sim_comb"], cand["sim_name"])
    cand = (cand.sort_values(["li", "_s"], ascending=[True, False])
                .groupby("li", sort=False).head(cfg.max_per_s1)
                .drop(columns=["_s", "rk_left", "rk_right", "rk_name"]).reset_index(drop=True))
    return cand
