"""Scalable candidate generation (blocking).

Two stages, both linear-ish in the number of records:

1. RETRIEVAL (wide, cheap, approximate) - builds a pool of plausible pairs:
   * dense ANN: every record -> sparse TF-IDF (char 3-grams of the core name +
     address words) -> TruncatedSVD (fit on a sample) -> L2-normalised dense
     vector -> FAISS HNSW index per country block. Queried in both directions:
       S1 -> S2/S3 (ann_k_left)  and  S2/S3 -> S1 (ann_k_right).
     Source 1 is deduplicated, so a S2/S3 record belongs to at most one S1;
     the reverse query recovers matches of S1 entities that have many duplicates.
   * exact keys (hash joins): identical core name, identical sorted name
     tokens, identical "house number + street word". Buckets bigger than
     key_bucket_cap are skipped, so this is O(N).
2. SELECTION (narrow, exact) - exact sparse cosine for every pooled pair, then a
   pair is kept only if it is among the top k_left of its S1 entity, the top
   k_right of its S2/S3 record, or the top k_name by name alone (or an exact
   name hit), passes absolute / relative similarity floors, and fits under the
   per-S1 cap. The survivors are candidate_pairs.tsv.

At billions of records the same design holds: FAISS indexes shard per country
(and per region if needed), key joins are map-reduce group-bys.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

try:
    import faiss
except ImportError:  # exact numpy fallback (fine for small data only)
    faiss = None


@dataclass
class BlockingConfig:
    # ---- retrieval --------------------------------------------------------
    dim: int = 128               # SVD dimensions of the dense vector
    svd_fit_rows: int = 300_000  # rows used to fit the SVD
    ann_k_left: int = 10         # ANN neighbours per S1 record
    ann_k_right: int = 3         # ANN neighbours per S2/S3 record
    hnsw_m: int = 32
    ef_search: int = 128
    exact_below: int = 20_000    # use exact (flat) search for blocks smaller than this
    key_bucket_cap: int = 20     # skip key buckets with more than this many S1 or S2/S3 records
    name_weight: float = 0.8     # weight of the name block in the combined vector
    # ---- selection ----------------------------------------------------------
    k_left: int = 2              # keep top-k per S1 entity (combined similarity)
    k_right: int = 1             # keep top-k per S2/S3 record
    k_name: int = 0              # keep top-k per S1 entity by name similarity only
    keep_exact_name: int = 0     # always keep exact core-name hits
    min_sim: float = 0.30        # absolute floor on combined cosine ...
    min_name_sim: float = 0.55   # ... unless the name alone is this similar
    rel_sim: float = 0.0         # keep only pairs with score >= rel_sim * best (per S1 or per S2/S3)
    max_per_s1: int = 5          # hard cap per Source-1 entity
    pair_chunk: int = 500_000    # pairs per chunk when computing exact cosines

    def to_dict(self):
        return asdict(self)


class Vectors:
    """TF-IDF matrices (fit on left+right) plus the dense ANN vectors."""

    def __init__(self, left: pd.DataFrame, right: pd.DataFrame, cfg: BlockingConfig, dense=True):
        self.nl = len(left)
        self.cfg = cfg
        names = pd.concat([left["name_core"], right["name_core"]], ignore_index=True)
        addrs = pd.concat([left["addr"] + " " + left["postal"], right["addr"] + " " + right["postal"]],
                          ignore_index=True)
        names = names.where(names != "", "_")
        addrs = addrs.where(addrs.str.strip() != "", "_")
        tf = dict(sublinear_tf=True, dtype=np.float32)
        Nc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, **tf).fit_transform(names)
        Nw = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", **tf).fit_transform(names)
        Aw = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", **tf).fit_transform(addrs)
        w = cfg.name_weight
        comb = normalize(sp.hstack([w * Nc, np.sqrt(1 - w * w) * Aw]).tocsr())
        self.mats = {"name_char": Nc.tocsr(), "name_word": Nw.tocsr(), "addr_word": Aw.tocsr(), "comb": comb}
        self.dense = self._svd(comb, cfg) if dense else None

    @staticmethod
    def _svd(X, cfg):
        n = X.shape[0]
        dim = max(2, min(cfg.dim, X.shape[1] - 1, n - 1))
        rng = np.random.default_rng(0)
        fit_idx = rng.choice(n, size=min(n, cfg.svd_fit_rows), replace=False)
        svd = TruncatedSVD(n_components=dim, algorithm="randomized", n_iter=5, random_state=0).fit(X[fit_idx])
        out = np.empty((n, dim), np.float32)
        for s in range(0, n, 500_000):
            out[s:s + 500_000] = svd.transform(X[s:s + 500_000])
        return normalize(out).astype(np.float32)

    def left(self, key):
        return self.mats[key][: self.nl]

    def right(self, key):
        return self.mats[key][self.nl:]

    def pair_cosine(self, key, li, ri):
        """Row-wise cosine for aligned index arrays (rows are L2-normalised)."""
        L, R = self.left(key), self.right(key)
        out = np.empty(len(li), np.float32)
        step = self.cfg.pair_chunk
        for s in range(0, len(li), step):
            a, b = L[li[s:s + step]], R[ri[s:s + step]]
            out[s:s + step] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
        return out


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------

def _knn(base: np.ndarray, query: np.ndarray, k: int, cfg: BlockingConfig):
    """Approximate inner-product kNN -> (query_row, base_row) arrays."""
    if len(base) == 0 or len(query) == 0 or k <= 0:
        e = np.empty(0, np.int64)
        return e, e
    k = min(k, len(base))
    d = base.shape[1]
    if faiss is not None:
        if len(base) < cfg.exact_below:
            index = faiss.IndexFlatIP(d)
        else:
            index = faiss.IndexHNSWFlat(d, cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 80
            index.hnsw.efSearch = max(cfg.ef_search, k)
        index.add(np.ascontiguousarray(base))
        _, I = index.search(np.ascontiguousarray(query), k)
    else:
        I = np.empty((len(query), k), np.int64)
        step = max(1, 20_000_000 // len(base))
        for s in range(0, len(query), step):
            S = query[s:s + step] @ base.T
            I[s:s + step] = np.argpartition(-S, k - 1, axis=1)[:, :k]
    q = np.repeat(np.arange(len(query)), k)
    b = I.ravel()
    ok = b >= 0
    return q[ok], b[ok]


def _country_groups(left, right):
    """Blocks by country label (open set). Records with an empty country join every block."""
    lc, rc = left["country_n"].values, right["country_n"].values
    countries = sorted((set(lc) | set(rc)) - {""})
    if not countries:
        yield np.arange(len(left)), np.arange(len(right))
        return
    for c in countries:
        yield np.where((lc == c) | (lc == ""))[0], np.where((rc == c) | (rc == ""))[0]


def _key_pairs(left, right, lkey, rkey, cap):
    a = pd.DataFrame({"k": lkey, "li": np.arange(len(left))})
    b = pd.DataFrame({"k": rkey, "ri": np.arange(len(right))})
    a = a[a["k"] != ""]
    b = b[b["k"] != ""]
    a = a[a["k"].map(a["k"].value_counts()) <= cap]
    b = b[b["k"].map(b["k"].value_counts()) <= cap]
    m = a.merge(b, on="k")
    return m["li"].values, m["ri"].values


def _name_keys(df):
    c = df["country_n"] + "|"
    nospace = df["name_core"].str.replace(" ", "", regex=False)
    sorted_tok = df["name_core"].map(lambda x: " ".join(sorted(x.split())))
    street = df["street_key"]
    return (c + nospace).where(nospace != "", ""), (c + sorted_tok).where(sorted_tok != "", ""), \
        (c + street).where(street != "", "")


def candidate_pool(left, right, vecs: Vectors, cfg: BlockingConfig) -> pd.DataFrame:
    """Wide pool of pairs with exact similarities: [li, ri, exact_name, sim_comb, sim_name]."""
    parts = []
    D = vecs.dense
    Dl, Dr = D[: vecs.nl], D[vecs.nl:]
    for gl, gr in _country_groups(left, right):
        if len(gl) == 0 or len(gr) == 0:
            continue
        q, b = _knn(Dr[gr], Dl[gl], cfg.ann_k_left, cfg)
        parts.append(pd.DataFrame({"li": gl[q], "ri": gr[b]}))
        q, b = _knn(Dl[gl], Dr[gr], cfg.ann_k_right, cfg)
        parts.append(pd.DataFrame({"li": gl[b], "ri": gr[q]}))

    lk, rk = _name_keys(left), _name_keys(right)
    exact_parts = []
    for j in range(3):
        li, ri = _key_pairs(left, right, lk[j].values, rk[j].values, cfg.key_bucket_cap)
        df = pd.DataFrame({"li": li, "ri": ri})
        parts.append(df)
        if j == 0:
            exact_parts.append(df)

    pool = pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)
    exact = exact_parts[0].assign(exact_name=1)
    pool = pool.merge(exact, on=["li", "ri"], how="left").fillna({"exact_name": 0})
    pool = pool.astype({"li": np.int64, "ri": np.int64, "exact_name": np.int8})
    pool["sim_comb"] = vecs.pair_cosine("comb", pool["li"].values, pool["ri"].values)
    pool["sim_name"] = vecs.pair_cosine("name_char", pool["li"].values, pool["ri"].values)
    return pool


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def select(pool: pd.DataFrame, cfg: BlockingConfig) -> pd.DataFrame:
    """Narrow the pool to the final candidate set (see module docstring)."""
    li, ri = pool["li"].values, pool["ri"].values
    sc = pd.Series(pool["sim_comb"].values)
    sn = pd.Series(pool["sim_name"].values)
    if "rk_l" not in pool:
        pool = add_ranks(pool)
    rk_l, rk_r, rk_n = pool["rk_l"].values, pool["rk_r"].values, pool["rk_n"].values
    keep = (rk_l <= cfg.k_left) | (rk_r <= cfg.k_right) | (rk_n <= cfg.k_name)
    if cfg.keep_exact_name:
        keep |= pool["exact_name"].values == 1
    keep &= (sc.values >= cfg.min_sim) | (sn.values >= cfg.min_name_sim)
    if cfg.rel_sim > 0:
        s = np.maximum(sc.values, sn.values)
        best_l = pd.Series(s).groupby(li).transform("max").values
        best_r = pd.Series(s).groupby(ri).transform("max").values
        keep &= (s >= cfg.rel_sim * best_l) | (s >= cfg.rel_sim * best_r)
    cand = pool[keep].copy()
    # cap order: pairs that are a top choice from either side first (a S2/S3 record's own best
    # S1 is very likely its true match), then by combined name+address similarity
    cand["_r"] = np.minimum(cand["rk_l"], cand["rk_r"])
    cand = (cand.sort_values(["li", "_r", "sim_comb"], ascending=[True, True, False])
                .groupby("li", sort=False).head(cfg.max_per_s1)
                .drop(columns=["_r", "rk_l", "rk_r", "rk_n"]).reset_index(drop=True))
    return cand


def add_ranks(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool.copy()
    li, ri = pool["li"].values, pool["ri"].values
    sc = pd.Series(pool["sim_comb"].values)
    sn = pd.Series(pool["sim_name"].values)
    pool["rk_l"] = sc.groupby(li).rank(ascending=False, method="first").values
    pool["rk_r"] = sc.groupby(ri).rank(ascending=False, method="first").values
    pool["rk_n"] = sn.groupby(li).rank(ascending=False, method="first").values
    return pool


def generate_candidates(left, right, vecs: Vectors, cfg: BlockingConfig) -> pd.DataFrame:
    return select(candidate_pool(left, right, vecs, cfg), cfg)
