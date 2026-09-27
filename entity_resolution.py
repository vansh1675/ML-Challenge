# =============================================================================
# Amazon ML Challenge - Business Entity Resolution (single-file version)
# Cells are separated by "# %%" (VS Code / Jupyter-compatible). Run top to bottom.
# pip install pandas numpy scipy scikit-learn lightgbm rapidfuzz faiss-cpu
# =============================================================================

# %% [IMPORTS]
import itertools
import json
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

try:
    import faiss
except ImportError:  # exact numpy fallback (fine for small data only)
    faiss = None


# %% [TEXT NORMALISATION]
# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

# Map every variant of a word onto one canonical short token.
NAME_CANON = {
    "corporation": "corp", "corpn": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "prv": "pvt",
    "company": "co", "co": "co", "compagnie": "cie", "cie": "cie",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc",
    "and": "and", "et": "and",
    "international": "intl", "intl": "intl", "internationale": "intl",
    "manufacturing": "mfg", "mfg": "mfg", "manufacturers": "mfg",
    "services": "svc", "service": "svc", "svcs": "svc", "svc": "svc",
    "technologies": "tech", "technology": "tech", "tech": "tech", "techn": "tech",
    "associates": "assoc", "association": "assoc", "assoc": "assoc", "assn": "assoc",
    "brothers": "bros", "bros": "bros", "bro": "bros",
    "national": "natl", "natl": "natl",
    "management": "mgmt", "mgmt": "mgmt", "mgt": "mgmt",
    "group": "grp", "grp": "grp", "groupe": "grp",
    "enterprises": "ent", "enterprise": "ent", "ent": "ent", "entp": "ent", "entreprise": "ent",
    "systems": "sys", "system": "sys", "sys": "sys",
    "solutions": "sol", "solution": "sol", "soln": "sol",
    "industries": "ind", "industry": "ind", "industrial": "ind", "inds": "ind", "ind": "ind",
    "hospital": "hosp", "hosp": "hosp",
    "university": "univ", "univ": "univ",
    "saint": "st", "st": "st", "ste": "st", "sainte": "st",
    "mount": "mt", "mt": "mt",
    "department": "dept", "dept": "dept",
    "centre": "ctr", "center": "ctr", "ctr": "ctr",
    "restaurant": "rest", "restaurants": "rest",
    "pharmacy": "pharm", "pharmaceuticals": "pharma", "pharma": "pharma",
    "societe": "soc", "society": "soc", "soc": "soc",
    "establishments": "ets", "etablissements": "ets", "ets": "ets",
    "trading": "trdg", "traders": "trdrs",
    "general": "gen", "gen": "gen",
    "construction": "const", "constructions": "const",
    "the": "the", "of": "of", "le": "le", "la": "la", "les": "les", "de": "de", "des": "des", "du": "du",
}

# Canonical tokens that carry (almost) no identity information.
NAME_STOP = {
    "inc", "ltd", "pvt", "corp", "co", "cie", "llc", "llp", "lp", "plc", "pllc", "pc",
    "and", "the", "of", "le", "la", "les", "de", "des", "du",
    "sa", "sas", "sarl", "eurl", "sasu", "snc", "sci", "gmbh", "ag", "bv", "nv", "opc",
    "india", "usa", "us", "america", "france",
}

ALIAS_SPLIT = re.compile(r"\b(?:dba|d b a|aka|a k a|t a|trading as|doing business as|formerly|fka|f k a)\b")


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _basic(s) -> str:
    if s is None or (isinstance(s, float) and s != s):
        return ""
    s = strip_accents(str(s)).lower()
    s = s.replace("&", " and ").replace("+", " and ").replace("@", " at ")
    s = re.sub(r"(?<=\w)['’`](?=\w)", "", s)          # o'neil -> oneil, mcdonald's -> mcdonalds
    s = re.sub(r"(?<=\b\w)\.(?=\w\b)", "", s)         # p.v.t / l.l.c -> pvt / llc
    s = re.sub(r"(?<=\b\w)\.(?=\w\.)", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def canon_name_tokens(s: str):
    return [NAME_CANON.get(t, t) for t in s.split()]


def norm_name(raw):
    """Return (full normalised name, core name without legal/stop tokens, list of alias names)."""
    b = _basic(raw)
    # merge split legal suffixes: "l l c" -> "llc", "p ltd" etc.
    b = re.sub(r"\bl l c\b", "llc", b)
    b = re.sub(r"\bl l p\b", "llp", b)
    b = re.sub(r"\bpvt ltd\b|\bprivate limited\b", "pvt ltd", b)
    toks = canon_name_tokens(b)
    full = " ".join(toks)
    core_toks = [t for t in toks if t not in NAME_STOP]
    core = " ".join(core_toks) if core_toks else full
    aliases = [p.strip() for p in ALIAS_SPLIT.split(full) if p.strip()]
    aliases = [" ".join(t for t in a.split() if t not in NAME_STOP) or a for a in aliases]
    return full, core, aliases


def acronym(core: str) -> str:
    return "".join(t[0] for t in core.split() if t and not t.isdigit())


# ---------------------------------------------------------------------------
# Address normalisation
# ---------------------------------------------------------------------------

ADDR_CANON = {
    "street": "st", "str": "st", "st": "st", "strasse": "st",
    "road": "rd", "rd": "rd", "marg": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "avn": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bld": "blvd", "boul": "blvd",
    "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln", "gali": "ln",
    "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl", "plaza": "plz", "plz": "plz",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "circle": "cir", "cir": "cir",
    "square": "sq", "sq": "sq",
    "terrace": "ter", "ter": "ter",
    "suite": "ste", "ste": "ste", "unit": "ste",
    "apartment": "apt", "apt": "apt", "flat": "apt",
    "floor": "fl", "fl": "fl", "flr": "fl",
    "building": "bldg", "bldg": "bldg", "bldng": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "near": "nr", "nr": "nr", "opposite": "opp", "opp": "opp", "behind": "bhd", "beside": "nr",
    "sector": "sec", "sec": "sec",
    "nagar": "ngr", "ngr": "ngr",
    "colony": "col", "col": "col",
    "post": "po", "po": "po",
    "district": "dist", "dist": "dist", "distt": "dist",
    "chemin": "ch", "ch": "ch", "route": "rte", "rte": "rte", "impasse": "imp", "imp": "imp",
    "allee": "all", "rue": "rue", "r": "rue",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft",
    "saint": "st", "sainte": "st",
    "office": "off", "off": "off", "shop": "shop", "plot": "plot", "gala": "shop",
    "cedex": "", "no": "", "number": "", "num": "", "nos": "",
}
ADDR_STOP = {"", "the", "and", "of"}  # keep "de"/"la" (= Delaware / Louisiana codes)

# Multi-word regions -> short codes (applied on the cleaned string before tokenising).
REGION_PHRASES = {
    # US states
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    # Indian states / UTs
    "andhra pradesh": "ap", "arunachal pradesh": "arp", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "chattisgarh": "cg", "goa": "goa", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn",
    "telangana": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "delhi", "new delhi": "delhi", "jammu and kashmir": "jk", "chandigarh": "chd",
    "puducherry": "py", "pondicherry": "py",
    # Indian city renames / transliterations
    "calcutta": "kolkata", "bombay": "mumbai", "madras": "chennai", "bangalore": "bengaluru",
    "bengalooru": "bengaluru", "poona": "pune", "gurgaon": "gurugram", "baroda": "vadodara",
    "trivandrum": "thiruvananthapuram", "cochin": "kochi", "mysore": "mysuru", "benares": "varanasi",
    "banaras": "varanasi", "allahabad": "prayagraj", "simla": "shimla", "calicut": "kozhikode",
    "vizag": "visakhapatnam", "mangalore": "mangaluru", "belgaum": "belagavi", "hubli": "hubballi",
    "cawnpore": "kanpur", "secunderabad": "hyderabad", "navi mumbai": "navimumbai",
}
_REGION_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, REGION_PHRASES), key=len, reverse=True)) + r")\b")

ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
}


def norm_address(raw):
    """Return (normalised address, number tokens, postal code or '', street key, region)."""
    if raw is None or (isinstance(raw, float) and raw != raw):
        return "", (), "", "", ""
    s = strip_accents(str(raw)).lower()
    s = re.sub(r"\b(\d{3})\s(\d{3})\b", r"\1\2", s)       # indian PIN "560 001"
    s = re.sub(r"\b(\d{2})\s(\d{3})\b", r"\1\2", s)       # french CP "75 008"
    s = re.sub(r"(\d+)(st|nd|rd|th|er|e|eme)\b", r"\1", s)  # 5th -> 5, 1er -> 1
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s)
    s = re.sub(r"([a-z])(\d)", r"\1 \2", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = _REGION_RE.sub(lambda m: REGION_PHRASES[m.group(1)], s)
    toks = []
    for t in s.split():
        t = ORDINAL_WORDS.get(t, t)
        t = ADDR_CANON.get(t, t)
        if t not in ADDR_STOP:
            toks.append(t)
    nums = tuple(t for t in toks if t.isdigit())
    postal = ""
    for t in reversed(nums):
        if len(t) in (5, 6):
            postal = t
            break
    # "house number + first street word" e.g. "720 lowell", robust to component reordering
    street = ""
    for a, b in zip(toks, toks[1:]):
        if a.isdigit() and a != postal and not b.isdigit() and len(b) > 1:
            street = f"{a} {b}"
            break
    alpha = [t for t in toks if not t.isdigit()]
    region = alpha[-1] if alpha else ""   # usually state code / city (last component)
    return " ".join(toks), nums, postal, street, region


def norm_country(raw) -> str:
    if raw is None or (isinstance(raw, float) and raw != raw):
        return ""
    return _basic(raw)


# %% [DATA LOADING]
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
    df["street_key"] = addrs.map(lambda x: x[3])
    df["region"] = addrs.map(lambda x: x[4])
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


# %% [BLOCKING]
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


# %% [FEATURES]
def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _tok_jaccard(a, b):
    out = np.zeros(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        u = len(sx | sy)
        out[i] = len(sx & sy) / u if u else 0.0
    return out


def _alias_best(al, ar, base):
    """Best token_set_ratio over DBA / aka alias pairs (falls back to the core-name score)."""
    out = base.copy()
    for i, (x, y) in enumerate(zip(al, ar)):
        if len(x) > 1 or len(y) > 1:
            out[i] = max(out[i], max(fuzz.token_set_ratio(p, q) for p in x for q in y))
    return out


def _tristate(a, b):
    """1 if both present and equal, 0 if both present and different, -1 if either missing."""
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    both = (a != "") & (b != "")
    return np.where(both, (a == b).astype(np.int8), -1).astype(np.int8)


def _num_feats(nl, nr):
    jac = np.full(len(nl), -1.0, np.float32)
    first = np.full(len(nl), -1, np.int8)
    for i, (x, y) in enumerate(zip(nl, nr)):
        if x and y:
            sx, sy = set(x), set(y)
            jac[i] = len(sx & sy) / len(sx | sy)
            first[i] = int(x[0] == y[0])
    return jac, first


def _name_nums(names):
    return ["".join(sorted(t for t in n.split() if t.isdigit())) for n in names]


def build_features(cand: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, vecs,
                   chunk: int = 1_000_000) -> pd.DataFrame:
    """Chunked so millions of pairs fit in memory; competition features need the full set."""
    parts = [_pair_features(cand.iloc[s:s + chunk], left, right, vecs) for s in range(0, len(cand), chunk)]
    F = pd.concat(parts, ignore_index=True) if parts else _pair_features(cand, left, right, vecs)
    return _competition_features(F, cand["li"].values, cand["ri"].values).astype(np.float32)


def _pair_features(cand, left, right, vecs) -> pd.DataFrame:
    li, ri = cand["li"].values, cand["ri"].values
    L, R = left.iloc[li], right.iloc[ri]
    ln, rn = L["name_core"].tolist(), R["name_core"].tolist()
    lf, rf = L["name_full"].tolist(), R["name_full"].tolist()
    la, ra = L["addr"].tolist(), R["addr"].tolist()

    F = pd.DataFrame(index=np.arange(len(cand)))
    # --- blocking similarities ------------------------------------------------
    F["sim_comb"] = cand["sim_comb"].values
    F["sim_name"] = cand["sim_name"].values
    F["exact_name"] = cand["exact_name"].values
    F["cos_name_word"] = vecs.pair_cosine("name_word", li, ri)
    F["cos_addr_word"] = vecs.pair_cosine("addr_word", li, ri)

    # --- name string similarity -----------------------------------------------
    F["n_ratio"] = _cp(ln, rn, fuzz.ratio)
    F["n_tset"] = _cp(ln, rn, fuzz.token_set_ratio)
    F["n_tsort"] = _cp(ln, rn, fuzz.token_sort_ratio)
    F["n_partial"] = _cp(ln, rn, fuzz.partial_ratio)
    F["n_jw"] = _cp(ln, rn, JaroWinkler.normalized_similarity)
    F["n_lev"] = _cp(ln, rn, Levenshtein.distance)
    F["nfull_ratio"] = _cp(lf, rf, fuzz.ratio)
    F["nfull_tset"] = _cp(lf, rf, fuzz.token_set_ratio)
    F["n_wratio"] = _cp(ln, rn, fuzz.WRatio)
    F["n_jacc"] = _tok_jaccard(ln, rn)
    F["n_alias"] = _alias_best(L["name_alias"].tolist(), R["name_alias"].tolist(), F["n_tset"].values)
    lt = [x.split()[0] if x else "" for x in ln]
    rt = [x.split()[0] if x else "" for x in rn]
    F["n_first_eq"] = np.array([a == b and a != "" for a, b in zip(lt, rt)], np.int8)
    lnc = [x.replace(" ", "") for x in ln]
    rnc = [x.replace(" ", "") for x in rn]
    F["n_nospace_eq"] = np.array([a == b for a, b in zip(lnc, rnc)], np.int8)
    la_ = L["name_acr"].tolist()
    ra_ = R["name_acr"].tolist()
    F["n_acronym"] = np.array([(len(a) >= 2 and a == y) or (len(b) >= 2 and b == x)
                               for a, b, x, y in zip(la_, ra_, lnc, rnc)], np.int8)
    F["n_num_eq"] = _tristate(_name_nums(ln), _name_nums(rn))
    ltok = np.array([len(x.split()) for x in ln], np.float32)
    rtok = np.array([len(x.split()) for x in rn], np.float32)
    F["n_len_l"] = ltok
    F["n_len_r"] = rtok
    F["n_len_diff"] = np.abs(ltok - rtok)

    # --- address similarity -----------------------------------------------------
    F["a_tset"] = _cp(la, ra, fuzz.token_set_ratio)
    F["a_tsort"] = _cp(la, ra, fuzz.token_sort_ratio)
    F["a_partial"] = _cp(la, ra, fuzz.partial_ratio)
    F["a_ratio"] = _cp(la, ra, fuzz.ratio)
    F["a_jacc"] = _tok_jaccard(la, ra)
    F["a_empty"] = ((L["addr"].values == "").astype(np.int8) + (R["addr"].values == "").astype(np.int8))
    F["a_postal"] = _tristate(L["postal"].values, R["postal"].values)
    F["a_street"] = _tristate(L["street_key"].values, R["street_key"].values)
    F["a_region"] = _tristate(L["region"].values, R["region"].values)
    jac, first = _num_feats(L["addr_nums"].tolist(), R["addr_nums"].tolist())
    F["a_num_jacc"] = jac
    F["a_num_first"] = first
    al = np.array([len(x.split()) for x in la], np.float32)
    ar = np.array([len(x.split()) for x in ra], np.float32)
    F["a_len_ratio"] = np.minimum(al, ar) / np.maximum(np.maximum(al, ar), 1)

    # --- context ------------------------------------------------------------------
    F["country_eq"] = _tristate(L["country_n"].values, R["country_n"].values)
    F["is_s3"] = (R["source"].values == "S3").astype(np.int8)

    return F


def _competition_features(F, g_l, g_r):
    """How does this pair compare with the rival candidates of the same S1 / same S2-S3 record?"""
    for col in ["sim_comb", "sim_name", "n_tset", "a_tset"]:
        v = pd.Series(F[col].values)
        F[f"{col}_gap_l"] = (v.groupby(g_l).transform("max") - v).values
        F[f"{col}_gap_r"] = (v.groupby(g_r).transform("max") - v).values
    F["rank_l"] = pd.Series(F["sim_comb"].values).groupby(g_l).rank(ascending=False, method="min").values
    F["rank_r"] = pd.Series(F["sim_comb"].values).groupby(g_r).rank(ascending=False, method="min").values
    F["ncand_l"] = pd.Series(g_l).map(pd.Series(g_l).value_counts()).values
    F["ncand_r"] = pd.Series(g_r).map(pd.Series(g_r).value_counts()).values
    return F


# %% [METRIC]
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


# %% [MODEL + DECISION RULE]
LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    seed=42,
)
NUM_ROUNDS = 600


def train_booster(X, y, rounds=NUM_ROUNDS, params=None):
    return lgb.train(params or LGB_PARAMS, lgb.Dataset(X, label=y), num_boost_round=rounds)


def oof_predict(X, y, groups, n_folds=5, rounds=NUM_ROUNDS):
    """Out-of-fold probabilities; folds are grouped by Source-1 entity so no entity leaks."""
    oof = np.zeros(len(y), np.float64)
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        b = train_booster(X.iloc[tr], y[tr], rounds)
        oof[va] = b.predict(X.iloc[va])
        print(f"  fold {k}: {len(va):,} pairs, pos={int(y[va].sum()):,}")
    return oof


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

def decide(li, ri, p, threshold, exclusive=True, rel=0.0):
    """Boolean mask of accepted pairs.

    exclusive: Source 1 is deduplicated, so an S2/S3 record can belong to at most
               one S1 entity -> keep only its highest-probability S1.
    rel:       additionally require p >= rel * (best p of that S1 entity).
    """
    p = np.asarray(p)
    keep = p >= threshold
    if exclusive:
        best_r = pd.Series(p).groupby(ri).transform("max").values
        keep &= p >= best_r
    if rel > 0:
        best_l = pd.Series(p).groupby(li).transform("max").values
        keep &= p >= rel * best_l
    return keep


def macro_f05_from_pairs(li, keep, label, n_true_per_left, beta=0.5):
    """Vectorised macro F0.5. n_true_per_left: array (len = #S1) of true match counts
    (counting positives that blocking missed, so recall is honest)."""
    n = len(n_true_per_left)
    tp = np.bincount(li, weights=(keep & (label == 1)).astype(float), minlength=n)
    npred = np.bincount(li, weights=keep.astype(float), minlength=n)
    ntrue = n_true_per_left.astype(float)
    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / npred, 0)
        rec = np.where(ntrue > 0, tp / ntrue, 0)
        f = np.where(tp > 0, (1 + b2) * prec * rec / (b2 * prec + rec), 0.0)
    f = np.where((ntrue == 0) & (npred == 0), 1.0, f)
    return float(f.mean())


def tune_decision(li, ri, p, label, n_true_per_left):
    """Grid-search the decision rule directly on macro F0.5 (the leaderboard metric)."""
    p = np.asarray(p)
    is_best_r = p >= pd.Series(p).groupby(ri).transform("max").values
    best_l = pd.Series(p).groupby(li).transform("max").values
    best = (-1, None)
    grid = itertools.product(np.round(np.arange(0.05, 0.96, 0.025), 3), [True, False], [0.0, 0.3, 0.6, 0.8])
    for t, ex, rel in grid:
        keep = (p >= t) & (is_best_r if ex else True) & (p >= rel * best_l)
        s = macro_f05_from_pairs(li, keep, label, n_true_per_left)
        if s > best[0]:
            best = (s, dict(threshold=float(t), exclusive=bool(ex), rel=float(rel)))
    return best


# %% [PIPELINE HELPERS]
def block_and_featurize(left, right, cfg: BlockingConfig):
    t0 = time.time()
    vecs = Vectors(left, right, cfg)
    print(f"vectors: {time.time() - t0:.1f}s")
    pool = candidate_pool(left, right, vecs, cfg)
    print(f"retrieval pool: {len(pool):,} pairs ({len(pool) / max(len(left), 1):.2f} per S1) "
          f"in {time.time() - t0:.1f}s")
    cand = select(pool, cfg)
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
    rows = [(i, ",".join(sorted(sets.get(i, set())))) for i in dict.fromkeys(order)]  # one row per S1 id
    pd.DataFrame(rows, columns=["source1_entity_id", col]).to_csv(path, sep="\t", index=False)


# %% [SUBMISSION VALIDATOR]
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


# %% [TRAIN]  -> OOF validation score + model saved to ARTIFACTS
DATA_DIR = "dataset"        # folder containing train/ and test/
ARTIFACTS = "artifacts"
OUT_DIR = "output"
N_FOLDS = 3
ROUNDS = NUM_ROUNDS
SAMPLE_S1 = 300_000         # train/tune on this many random S1 entities (0 = all); blocking uses all data
CFG = BlockingConfig()      # e.g. BlockingConfig(k_left=2, k_right=1, max_per_s1=5) after tuning


def train(data_dir=DATA_DIR, artifacts=ARTIFACTS, cfg=CFG, n_folds=N_FOLDS, rounds=ROUNDS, sample_s1=SAMPLE_S1):
    os.makedirs(artifacts, exist_ok=True)
    left, right = load_split(data_dir, "train")
    truth = load_ground_truth(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
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

    if sample_s1 and sample_s1 < len(left):
        chosen = np.zeros(len(left), bool)
        chosen[np.random.default_rng(0).choice(len(left), sample_s1, replace=False)] = True
        m = chosen[li]
        X, y, li, ri = X[m].reset_index(drop=True), y[m], li[m], ri[m]
        remap = -np.ones(len(left), np.int64)
        remap[chosen] = np.arange(chosen.sum())
        li_eval, n_true_eval = remap[li], n_true[chosen]
        ids_eval = [i for i, c in zip(ids, chosen) if c]
        print(f"training on {chosen.sum():,} sampled S1 entities / {len(y):,} pairs")
    else:
        li_eval, n_true_eval, ids_eval = li, n_true, ids

    print("out-of-fold training ...")
    oof = oof_predict(X, y, groups=li, n_folds=n_folds, rounds=rounds)
    score, dec = tune_decision(li_eval, ri, oof, y, n_true_eval)
    print(f"OOF macro F0.5 = {score:.4f} with {dec}")
    pred = to_sets(left, right, li, ri, decide(li, ri, oof, **dec))
    country = dict(zip(ids, left["country_n"]))
    for c in sorted(set(left["country_n"])):
        sub = [i for i in ids_eval if country[i] == c]
        if sub:
            print(f"  country={c or '<empty>'}: {macro_f05(pred, truth, sub):.4f} ({len(sub):,} S1)")

    print("fitting final model on all training pairs ...")
    booster = train_booster(X, y, rounds)
    booster.save_model(os.path.join(artifacts, "model.txt"))
    imp = pd.Series(booster.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print("top features:\n" + imp.head(15).round(0).to_string())
    with open(os.path.join(artifacts, "config.json"), "w") as f:
        json.dump({"blocking": cfg.to_dict(), "decision": dec, "features": list(X.columns),
                   "oof_macro_f05": score, "blocking_report": rep}, f, indent=2)
    print(f"saved to {artifacts}/")


# %% [PREDICT]  -> output/matching_results.tsv + output/candidate_pairs.tsv
def predict(data_dir=DATA_DIR, split="test", artifacts=ARTIFACTS, out_dir=OUT_DIR):
    with open(os.path.join(artifacts, "config.json")) as f:
        conf = json.load(f)
    cfg = BlockingConfig(**conf["blocking"])
    booster = lgb.Booster(model_file=os.path.join(artifacts, "model.txt"))
    left, right = load_split(data_dir, split)
    cand, X = block_and_featurize(left, right, cfg)
    li, ri = cand["li"].values, cand["ri"].values
    p = booster.predict(X[conf["features"]])
    keep = decide(li, ri, p, **conf["decision"])

    order = left["entity_id"].tolist()
    os.makedirs(out_dir, exist_ok=True)
    cand_path = os.path.join(out_dir, "candidate_pairs.tsv")
    match_path = os.path.join(out_dir, "matching_results.tsv")
    matches = to_sets(left, right, li, ri, keep)
    write_id_lists(to_sets(left, right, li, ri), order, "candidate_entity_ids", cand_path)
    write_id_lists(matches, order, "matched_entity_ids", match_path)
    print(f"wrote {match_path}: {int(keep.sum()):,} links, "
          f"{np.mean([len(s) == 0 for s in matches.values()]):.1%} empty rows")
    print(f"wrote {cand_path}: {len(cand):,} pairs ({len(cand) / max(len(left), 1):.2f} per S1)")
    validate(data_dir, split, match_path, cand_path)


# %% [RUN]
if __name__ == "__main__":
    train()
    predict()
