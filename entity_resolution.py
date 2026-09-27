# =============================================================================
# Amazon ML Challenge - Business Entity Resolution
# LOW-MEMORY STREAMING VERSION (runs on an 8 GB laptop with ~1.6M S1 + ~10M S2/S3 rows)
#
# Idea: Source 1 (deduplicated, smallest) is indexed ONCE. Source 2/3 are read in
# chunks and each chunk is matched against the S1 index, so the 10M records are
# never all in memory. Pass 1 builds the candidate pool, pass 2 computes features
# only for the final candidates. Normalised chunks are cached on disk (WORK_DIR).
#
# Cells are separated by "# %%". Run top to bottom.
# pip install pandas numpy scipy scikit-learn lightgbm rapidfuzz faiss-cpu psutil
# =============================================================================

# %% [IMPORTS]
import gc
import glob
import itertools
import os
import json
import re
import sys
import time
import unicodedata
import zlib
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

try:
    import psutil
except ImportError:
    psutil = None


def mem():
    """RAM used by this Python process now / at its peak so far (for progress logs)."""
    if psutil is None:
        return ""
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)          # Windows reports the peak directly
    if peak is None:
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except ImportError:
            peak = info.rss
    return f"[RAM {info.rss / 2**30:.1f} GB, peak {peak / 2**30:.1f} GB]"


print("faiss available:", faiss is not None)


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
    addr = " ".join(toks)
    return (addr,) + addr_parts(addr)


def addr_parts(addr):
    """(number tokens, postal code or '', street key, region) from a normalised address string."""
    toks = addr.split()
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
    return nums, postal, street, region


def norm_country(raw) -> str:
    if raw is None or (isinstance(raw, float) and raw != raw):
        return ""
    return _basic(raw)




# %% [SETTINGS]  <-- the only cell you need to edit
BASE = r"C:\Users\bansa\Downloads\6ab10eb3b23ba_student_resource\student_resource"

DATA_DIR = os.path.join(BASE, "dataset")      # contains train\ and test\
WORK_DIR = os.path.join(BASE, "work")         # disk cache of normalised S2/S3 chunks (safe to delete)
ARTIFACTS = os.path.join(BASE, "artifacts")   # model.txt + config.json
OUT_DIR = os.path.join(BASE, "output")        # matching_results.tsv + candidate_pairs.tsv

TRAIN_FRAC = 0.5      # fraction of training S1 entities (+ their matches) to use. 0.05 = quick test, 1.0 = all
CHUNK_ROWS = 200_000  # S2/S3 rows processed at a time. Lower (100_000) if you still run out of memory
N_FOLDS = 3
ROUNDS = 600          # LightGBM rounds; 200 for a quick test
MAX_TRAIN_PAIRS = 3_000_000   # cap on pairs fed to LightGBM (random S1 entities) to bound RAM


# %% [IO + COMPACT PREPROCESSING]
def read_tsv(path, chunksize=None, usecols=None):
    # keep_default_na=False: empty fields stay "" (and a business called "NA" survives)
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[], quoting=3,
                       chunksize=chunksize, usecols=usecols)


def _hash01(ids):
    """Deterministic pseudo-random number in [0,1) per id (for reproducible sampling)."""
    return np.array([zlib.crc32(s.encode()) for s in ids], dtype=np.float64) / 2 ** 32


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw rows -> 6 compact normalised columns (raw text dropped; postal / street / region are
    re-derived from `addr` only when needed, to save RAM)."""
    names = [norm_name(x) for x in df["business_name"].tolist()]
    return pd.DataFrame({
        "entity_id": df["entity_id"].tolist(),
        "country_n": [sys.intern(norm_country(x)) for x in df["country"].tolist()],
        "name_full": [n[0] for n in names],
        "name_core": [n[1] for n in names],
        # aliases only stored when a DBA / aka split exists
        "name_alias": ["|".join(n[2]) if len(n[2]) > 1 else "" for n in names],
        "addr": [norm_address(x)[0] for x in df["business_address"].tolist()],
    })


def load_truth(path):
    """Compact ground truth: {S2/S3 id -> S1 id} plus {S1 id -> number of true matches}."""
    r2s1, n_true = {}, {}
    for ch in read_tsv(path, chunksize=500_000):
        for s1, ids in zip(ch["source1_entity_id"].tolist(), ch["matched_entity_ids"].tolist()):
            ms = [x.strip() for x in ids.split(",") if x.strip()]
            n_true[s1] = len(ms)
            for m in ms:
                r2s1[m] = s1
    return r2s1, n_true


def load_s1(data_dir, split, frac=1.0):
    df = read_tsv(os.path.join(data_dir, split, f"{split}_source1.tsv"))
    df = df.drop_duplicates("entity_id")
    if frac < 1.0:
        df = df[_hash01(df["entity_id"].tolist()) < frac]
    s1 = prepare(df).reset_index(drop=True)
    del df
    gc.collect()
    return s1


def iter_right_raw(data_dir, split, chunk_rows):
    for src in (2, 3):
        path = os.path.join(data_dir, split, f"{split}_source{src}.tsv")
        for ch in read_tsv(path, chunksize=chunk_rows):
            yield ch


# %% [BLOCKING CONFIG + S1 INDEX]
@dataclass
class BlockingConfig:
    # ---- retrieval (S2/S3 record -> nearest S1 entities) --------------------
    dim: int = 64                 # SVD dimensions of the dense vector (64 keeps RAM low)
    fit_rows: int = 300_000       # S1 rows used to fit TF-IDF + SVD (a sample keeps RAM low)
    ann_k: int = 5                # ANN neighbours (S1 entities) retrieved per S2/S3 record
    pool_k: int = 3               # after exact rescoring keep this many S1 per S2/S3 record
    hnsw_m: int = 16
    ef_search: int = 64
    exact_below: int = 20_000     # exact (flat) search when a country block is smaller than this
    key_bucket_cap: int = 20      # ignore exact-key buckets with more S1 records than this
    name_weight: float = 0.8      # sim_comb = w^2 * name_cos + (1 - w^2) * addr_cos
    # ---- selection (final candidate set) --------------------------------------
    k_left: int = 2               # keep top-k S2/S3 records per S1 entity
    k_right: int = 1              # keep top-k S1 entities per S2/S3 record
    k_name: int = 0               # keep top-k per S1 entity by name similarity only
    keep_exact_name: int = 0      # always keep exact core-name hits
    min_sim: float = 0.30         # absolute floor on combined similarity ...
    min_name_sim: float = 0.55    # ... unless the name alone is this similar
    max_per_s1: int = 5           # hard cap per S1 entity

    def to_dict(self):
        return asdict(self)


class S1Index:
    """Everything about Source 1 needed to score S2/S3 chunks: TF-IDF models (fit on S1 only),
    S1 sparse vectors, SVD, one FAISS index per country, and exact-key lookup tables."""

    def __init__(self, s1: pd.DataFrame, cfg: BlockingConfig):
        t0 = time.time()
        self.cfg = cfg
        n = len(s1)
        # fit TF-IDF and SVD on a random sample of S1 (IDF / n-gram vocabulary barely change)
        fit_idx = np.sort(np.random.default_rng(0).choice(n, size=min(n, cfg.fit_rows), replace=False))
        sample = s1.iloc[fit_idx]
        names = sample["name_core"].where(sample["name_core"] != "", "_")
        addrs = sample["addr"].where(sample["addr"] != "", "_")
        kw = dict(sublinear_tf=True, dtype=np.float32)
        self.v_nc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, **kw).fit(names)
        self.v_nw = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=2, **kw).fit(names)
        self.v_aw = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=2, **kw).fit(addrs)
        Nc, _, Aw = self._sparse(sample)
        comb = self._comb(Nc, Aw)
        dim = max(2, min(cfg.dim, comb.shape[1] - 1, len(fit_idx) - 1))
        self.svd = TruncatedSVD(n_components=dim, algorithm="randomized", n_iter=4, random_state=0).fit(comb)
        del sample, names, addrs, Nc, Aw, comb
        # transform all S1 in chunks
        parts, dense = [], np.empty((n, dim), np.float32)
        for a in range(0, n, 200_000):
            Nc, Nw, Aw = self._sparse(s1.iloc[a:a + 200_000])
            dense[a:a + 200_000] = self._dense(self._comb(Nc, Aw))
            parts.append((Nc, Nw, Aw))
        self.Nc = sp.vstack([x[0] for x in parts]).tocsr()
        self.Nw = sp.vstack([x[1] for x in parts]).tocsr()
        self.Aw = sp.vstack([x[2] for x in parts]).tocsr()
        del parts
        gc.collect()
        print(f"  S1 vectors ready in {time.time() - t0:.0f}s {mem()}")

        # one ANN index per country (S1 rows with an empty country go into every index)
        c = s1["country_n"].to_numpy(dtype=object)
        self.countries = sorted(set(c) - {""}) or [""]
        self.index, self.members = {}, {}
        for cc in self.countries:
            rows = np.where((c == cc) | (c == ""))[0] if cc else np.arange(n)
            self.members[cc] = rows
            self.index[cc] = self._build(dense[rows])
        del dense
        gc.collect()

        # exact-key lookup tables: key -> array of S1 rows (small buckets only)
        self.keys = [self._key_table(k) for k in _keys(s1)]
        self.name_hash = _h64([x.replace(" ", "") for x in s1["name_core"].tolist()])
        print(f"  S1 index ready ({n:,} entities, countries={self.countries}) in {time.time() - t0:.0f}s {mem()}")

    def _sparse(self, df):
        names = df["name_core"].where(df["name_core"] != "", "_")
        addrs = df["addr"].where(df["addr"] != "", "_")
        return (normalize(self.v_nc.transform(names)).tocsr(), normalize(self.v_nw.transform(names)).tocsr(),
                normalize(self.v_aw.transform(addrs)).tocsr())

    def _comb(self, Nc, Aw):
        w = self.cfg.name_weight
        return sp.hstack([w * Nc, np.sqrt(1 - w * w) * Aw]).tocsr()

    def _dense(self, comb):
        return normalize(self.svd.transform(comb)).astype(np.float32)

    def chunk_vectors(self, rdf):
        Nc, Nw, Aw = self._sparse(rdf)
        return Nc, Nw, Aw, self._dense(self._comb(Nc, Aw))

    def _build(self, X):
        d = X.shape[1]
        if faiss is None:
            return X
        if len(X) < self.cfg.exact_below:
            idx = faiss.IndexFlatIP(d)
        else:
            idx = faiss.IndexHNSWFlat(d, self.cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = 64
            idx.hnsw.efSearch = max(self.cfg.ef_search, self.cfg.ann_k)
        idx.add(np.ascontiguousarray(X))
        return idx

    def search(self, cc, Q, k):
        """-> (query_row, s1_row) arrays."""
        members = self.members[cc]
        k = min(k, len(members))
        if len(Q) == 0 or k == 0:
            e = np.empty(0, np.int64)
            return e, e
        idx = self.index[cc]
        if faiss is not None:
            _, I = idx.search(np.ascontiguousarray(Q), k)
        else:
            S = Q @ idx.T
            I = np.argpartition(-S, k - 1, axis=1)[:, :k]
        q = np.repeat(np.arange(len(Q)), k)
        b = I.ravel()
        ok = b >= 0
        return q[ok], members[b[ok]]

    def _key_table(self, keys):
        """Sorted (hash, S1 row) arrays; buckets larger than key_bucket_cap are dropped."""
        rows = np.arange(len(keys))
        ok = keys != 0
        keys, rows = keys[ok], rows[ok]
        uniq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
        ok = cnt[inv] <= self.cfg.key_bucket_cap
        keys, rows = keys[ok], rows[ok]
        order = np.argsort(keys, kind="stable")
        return keys[order], rows[order].astype(np.int32)

    @staticmethod
    def key_lookup(table, qkeys):
        """-> (query_row, s1_row) for every query key found in the table."""
        tk, tr = table
        a = np.searchsorted(tk, qkeys, side="left")
        b = np.searchsorted(tk, qkeys, side="right")
        n = np.where(qkeys != 0, b - a, 0)
        q = np.repeat(np.arange(len(qkeys)), n)
        starts = np.repeat(a, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        return q, tr[starts].astype(np.int64)


def _h64(strings):
    """Strings -> uint64 hashes ('' -> 0), so key tables are small numpy arrays, not dicts."""
    h = pd.util.hash_array(np.asarray(strings, dtype=object))
    h[np.asarray([x == "" for x in strings])] = 0
    return h


def _keys(df):
    """Exact blocking keys (hashed): country|name-without-spaces, country|sorted name tokens,
    country|house#+street."""
    c = [x + "|" for x in df["country_n"].tolist()]
    core = df["name_core"].tolist()
    street = [addr_parts(a)[2] for a in df["addr"].tolist()]
    k1 = [cc + x.replace(" ", "") if x else "" for cc, x in zip(c, core)]
    k2 = [cc + " ".join(sorted(x.split())) if x else "" for cc, x in zip(c, core)]
    k3 = [cc + x if x else "" for cc, x in zip(c, street)]
    return _h64(k1), _h64(k2), _h64(k3)


def rowdot(A, ia, B, ib, step=100_000):
    """Cosine of rows A[ia[k]] and B[ib[k]] (rows are L2-normalised), computed in slices."""
    out = np.empty(len(ia), np.float32)
    for s in range(0, len(ia), step):
        out[s:s + step] = np.asarray(A[ia[s:s + step]].multiply(B[ib[s:s + step]]).sum(axis=1)).ravel()
    return out


# %% [PASS 1: STREAM S2/S3 -> CANDIDATE POOL -> FINAL CANDIDATES]
def build_pool(s1idx: S1Index, data_dir, split, work_dir, cfg: BlockingConfig, keep_right=None):
    """Streams S2/S3, caches each normalised chunk to disk, returns the candidate pool
    (li = S1 row, ri = global S2/S3 row) and the list of cached chunk files."""
    os.makedirs(work_dir, exist_ok=True)
    for f in glob.glob(os.path.join(work_dir, f"{split}_chunk_*.pkl")):
        os.remove(f)
    known_c = set(s1idx.countries)
    w2 = cfg.name_weight ** 2
    pool_parts, chunk_files, offset, t0 = [], [], 0, time.time()
    for k, raw in enumerate(iter_right_raw(data_dir, split, CHUNK_ROWS)):
        if keep_right is not None:
            raw = raw[keep_right(raw["entity_id"].tolist())]
        if len(raw) == 0:
            continue
        rdf = prepare(raw)
        del raw
        path = os.path.join(work_dir, f"{split}_chunk_{k:04d}.pkl")
        rdf.to_pickle(path)
        chunk_files.append((path, offset, len(rdf)))

        Nc, Nw, Aw, dense = s1idx.chunk_vectors(rdf)
        rc = rdf["country_n"].to_numpy(dtype=object)
        qs, ls = [], []
        for cc in s1idx.countries:
            rows = np.arange(len(rdf)) if cc == "" else \
                np.where((rc == cc) | (rc == "") | ~np.isin(rc, list(known_c)))[0]
            q, l = s1idx.search(cc, dense[rows], cfg.ann_k)
            qs.append(rows[q])
            ls.append(l)
        for table, keys in zip(s1idx.keys, _keys(rdf)):          # exact-key channel
            q, l = S1Index.key_lookup(table, keys)
            qs.append(q)
            ls.append(l)
        p = pd.DataFrame({"r": np.concatenate(qs).astype(np.int64),
                          "li": np.concatenate(ls).astype(np.int64)}).drop_duplicates()
        r, li = p["r"].values, p["li"].values
        sim_name = rowdot(s1idx.Nc, li, Nc, r)
        sim_comb = w2 * sim_name + (1 - w2) * rowdot(s1idx.Aw, li, Aw, r)
        rh = _h64([x.replace(" ", "") for x in rdf["name_core"].tolist()])
        exact = (s1idx.name_hash[li] == rh[r]) & (rh[r] != 0)
        p = pd.DataFrame({"li": li.astype(np.int32), "ri": (r + offset).astype(np.int32),
                          "sim_comb": sim_comb, "sim_name": sim_name, "exact_name": exact.astype(np.int8)})
        p = p[(p["sim_comb"] >= cfg.min_sim) | (p["sim_name"] >= cfg.min_name_sim)]
        # keep only the best pool_k S1 entities for each S2/S3 record (+ exact name hits).
        # Every S2/S3 record lives in exactly one chunk, so its rank here is final.
        p["rk_r"] = p.groupby("ri")["sim_comb"].rank(ascending=False, method="first").astype(np.int16)
        p = p[(p["rk_r"] <= cfg.pool_k) | (p["exact_name"] == 1)]
        pool_parts.append(p)
        offset += len(rdf)
        if len(pool_parts) >= 5:   # prune the accumulated pool so RAM stays bounded
            pool_parts = [_prune(pd.concat(pool_parts, ignore_index=True), cfg)]
        del rdf, Nc, Nw, Aw, dense
        gc.collect()
        if k % 5 == 0:
            n_pool = sum(len(x) for x in pool_parts)
            print(f"  chunk {k}: {offset:,} S2/S3 rows streamed, pool={n_pool:,} pairs, "
                  f"{time.time() - t0:.0f}s {mem()}")
    pool = _prune(pd.concat(pool_parts, ignore_index=True), cfg)
    print(f"pass 1 done: {offset:,} S2/S3 rows, pool={len(pool):,} pairs in {time.time() - t0:.0f}s {mem()}")
    return pool, chunk_files


def _prune(pool, cfg):
    """Drop pairs that can no longer reach the final candidate set. Safe to run repeatedly:
    a pair outside the top-k of its S1 now can only fall further as more chunks arrive."""
    li = pool["li"].values
    keep = pool["rk_r"].values <= cfg.k_right
    keep |= pd.Series(pool["sim_comb"].values).groupby(li).rank(ascending=False, method="first").values \
        <= cfg.k_left
    if cfg.k_name > 0:
        keep |= pd.Series(pool["sim_name"].values).groupby(li).rank(ascending=False, method="first").values \
            <= cfg.k_name
    if cfg.keep_exact_name:
        keep |= pool["exact_name"].values == 1
    return pool[keep].reset_index(drop=True)


def select(pool: pd.DataFrame, cfg: BlockingConfig) -> pd.DataFrame:
    """Final candidate set = what the model scores = candidate_pairs.tsv."""
    li = pool["li"].values
    sc = pd.Series(pool["sim_comb"].values)
    rk_l = sc.groupby(li).rank(ascending=False, method="first").values
    rk_r = pool["rk_r"].values        # rank among ALL S1 candidates of this S2/S3 record (from pass 1)
    keep = (rk_l <= cfg.k_left) | (rk_r <= cfg.k_right)
    if cfg.k_name > 0:
        rk_n = pd.Series(pool["sim_name"].values).groupby(li).rank(ascending=False, method="first").values
        keep |= rk_n <= cfg.k_name
    if cfg.keep_exact_name:
        keep |= pool["exact_name"].values == 1
    cand = pool[keep].copy()
    # cap order: pairs that are a top choice from either side first, then by similarity
    cand["_r"] = np.minimum(rk_l[keep], rk_r[keep])
    cand = (cand.sort_values(["li", "_r", "sim_comb"], ascending=[True, True, False])
                .groupby("li", sort=False).head(cfg.max_per_s1)
                .drop(columns=["_r", "rk_r"]))
    return cand.sort_values("ri").reset_index(drop=True)   # sorted by ri -> matches chunk order in pass 2


# %% [PASS 2: FEATURES FOR FINAL CANDIDATES]
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
        if "|" in x or "|" in y:
            xs, ys = x.split("|"), y.split("|")
            out[i] = max(out[i], max(fuzz.token_set_ratio(p, q) for p in xs for q in ys))
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


def pair_features(cand, s1, s1idx, rdf, r_local, rvec):
    """Features for pairs (S1 row cand.li, S2/S3 chunk row r_local)."""
    li = cand["li"].values
    Nw, Aw = rvec
    L, R = s1.iloc[li], rdf.iloc[r_local]
    ln, rn = L["name_core"].tolist(), R["name_core"].tolist()
    lf, rf = L["name_full"].tolist(), R["name_full"].tolist()
    la, ra = L["addr"].tolist(), R["addr"].tolist()

    F = pd.DataFrame(index=np.arange(len(cand)))
    F["sim_comb"] = cand["sim_comb"].values
    F["sim_name"] = cand["sim_name"].values
    F["exact_name"] = cand["exact_name"].values
    F["cos_name_word"] = rowdot(s1idx.Nw, li, Nw, r_local)
    F["cos_addr_word"] = rowdot(s1idx.Aw, li, Aw, r_local)
    # name
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
    al = [a or c for a, c in zip(L["name_alias"].tolist(), ln)]   # "" = no alias -> core name
    ar = [a or c for a, c in zip(R["name_alias"].tolist(), rn)]
    F["n_alias"] = _alias_best(al, ar, F["n_tset"].values)
    lt = [x.split()[0] if x else "" for x in ln]
    rt = [x.split()[0] if x else "" for x in rn]
    F["n_first_eq"] = np.array([a == b and a != "" for a, b in zip(lt, rt)], np.int8)
    lnc = [x.replace(" ", "") for x in ln]
    rnc = [x.replace(" ", "") for x in rn]
    F["n_nospace_eq"] = np.array([a == b for a, b in zip(lnc, rnc)], np.int8)
    la_, ra_ = [acronym(x) for x in ln], [acronym(x) for x in rn]
    F["n_acronym"] = np.array([(len(a) >= 2 and a == y) or (len(b) >= 2 and b == x)
                               for a, b, x, y in zip(la_, ra_, lnc, rnc)], np.int8)
    F["n_num_eq"] = _tristate(_name_nums(ln), _name_nums(rn))
    ltok = np.array([len(x.split()) for x in ln], np.float32)
    rtok = np.array([len(x.split()) for x in rn], np.float32)
    F["n_len_l"], F["n_len_r"], F["n_len_diff"] = ltok, rtok, np.abs(ltok - rtok)
    # address
    F["a_tset"] = _cp(la, ra, fuzz.token_set_ratio)
    F["a_tsort"] = _cp(la, ra, fuzz.token_sort_ratio)
    F["a_partial"] = _cp(la, ra, fuzz.partial_ratio)
    F["a_ratio"] = _cp(la, ra, fuzz.ratio)
    F["a_jacc"] = _tok_jaccard(la, ra)
    F["a_empty"] = (np.array([x == "" for x in la], np.int8) + np.array([x == "" for x in ra], np.int8))
    pl = [addr_parts(a) for a in la]     # (numbers, postal, street key, region)
    pr = [addr_parts(a) for a in ra]
    F["a_postal"] = _tristate([x[1] for x in pl], [x[1] for x in pr])
    F["a_street"] = _tristate([x[2] for x in pl], [x[2] for x in pr])
    F["a_region"] = _tristate([x[3] for x in pl], [x[3] for x in pr])
    jac, first = _num_feats([x[0] for x in pl], [x[0] for x in pr])
    F["a_num_jacc"], F["a_num_first"] = jac, first
    al = np.array([len(x.split()) for x in la], np.float32)
    ar = np.array([len(x.split()) for x in ra], np.float32)
    F["a_len_ratio"] = np.minimum(al, ar) / np.maximum(np.maximum(al, ar), 1)
    # context
    F["country_eq"] = _tristate(L["country_n"].values, R["country_n"].values)
    F["is_s3"] = np.array([x.startswith("S3") for x in R["entity_id"].tolist()], np.int8)
    return F.astype(np.float32)


COMP_BASE = ["sim_comb", "sim_name", "n_tset", "a_tset"]
COMP_COLS = [f"{c}_gap_{side}" for c in COMP_BASE for side in "lr"] + ["rank_l", "rank_r", "ncand_l", "ncand_r"]


def competition_features(X, cols, g_l, g_r):
    """How does this pair compare with rival candidates of the same S1 / same S2-S3 record?
    Written in place into the preallocated float32 matrix X."""
    j = {c: i for i, c in enumerate(cols)}
    for c in COMP_BASE:
        v = pd.Series(X[:, j[c]])
        X[:, j[f"{c}_gap_l"]] = v.groupby(g_l).transform("max").values - v.values
        X[:, j[f"{c}_gap_r"]] = v.groupby(g_r).transform("max").values - v.values
    sc = pd.Series(X[:, j["sim_comb"]])
    X[:, j["rank_l"]] = sc.groupby(g_l).rank(ascending=False, method="min").values
    X[:, j["rank_r"]] = sc.groupby(g_r).rank(ascending=False, method="min").values
    X[:, j["ncand_l"]] = pd.Series(g_l).map(pd.Series(g_l).value_counts()).values
    X[:, j["ncand_r"]] = pd.Series(g_r).map(pd.Series(g_r).value_counts()).values


def featurize(cand, s1, s1idx, chunk_files):
    """Pass 2: re-reads the cached chunks and computes features only for final candidates, straight
    into one preallocated float32 matrix (no concatenation copies).
    Returns features X and the S2/S3 entity id of every candidate."""
    t0 = time.time()
    if len(cand) == 0:
        raise ValueError("no candidate pairs - check the data paths / blocking settings")
    ri = cand["ri"].values
    X, cols, nb = None, None, 0
    rid = np.empty(len(cand), dtype=object)
    for path, off, n in chunk_files:
        a, b = np.searchsorted(ri, [off, off + n])
        if a == b:
            continue
        rdf = pd.read_pickle(path)
        sub = cand.iloc[a:b]
        r_local = sub["ri"].values - off
        _, Nw, Aw = s1idx._sparse(rdf)
        F = pair_features(sub, s1, s1idx, rdf, r_local, (Nw, Aw))
        if X is None:
            nb = F.shape[1]
            cols = list(F.columns) + COMP_COLS
            X = np.empty((len(cand), len(cols)), np.float32)
        X[a:b, :nb] = F.values
        rid[a:b] = rdf["entity_id"].values[r_local]
        del rdf, F
        gc.collect()
    competition_features(X, cols, cand["li"].values, cand["ri"].values)
    X = pd.DataFrame(X, columns=cols, copy=False)
    print(f"pass 2 done: {X.shape[1]} features for {len(X):,} pairs in {time.time() - t0:.0f}s {mem()}")
    return X, rid


def run_blocking(data_dir, split, cfg, s1, keep_right=None):
    t0 = time.time()
    print(f"[{split}] building S1 index ... {mem()}")
    s1idx = S1Index(s1, cfg)
    work = os.path.join(WORK_DIR, split)
    pool, chunk_files = build_pool(s1idx, data_dir, split, work, cfg, keep_right)
    cand = select(pool, cfg)
    del pool
    gc.collect()
    print(f"[{split}] final candidates: {len(cand):,} pairs = {len(cand) / max(len(s1), 1):.2f} per S1 "
          f"({time.time() - t0:.0f}s) {mem()}")
    X, rid = featurize(cand, s1, s1idx, chunk_files)
    del s1idx
    gc.collect()
    return cand, X, rid


# %% [METRIC + MODEL + DECISION RULE]
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=20,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=127, verbose=-1, seed=42)


def per_entity_f05(li, keep, label, n_true, beta=0.5):
    """F0.5 of every S1 entity (singletons included): 1.0 for a correct empty prediction."""
    n = len(n_true)
    tp = np.bincount(li, weights=(keep & (label == 1)).astype(float), minlength=n)
    npred = np.bincount(li, weights=keep.astype(float), minlength=n)
    ntrue = n_true.astype(float)
    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / npred, 0)
        rec = np.where(ntrue > 0, tp / ntrue, 0)
        f = np.where(tp > 0, (1 + b2) * prec * rec / (b2 * prec + rec), 0.0)
    return np.where((ntrue == 0) & (npred == 0), 1.0, f)


def decide(li, ri, p, threshold, exclusive=True, rel=0.0):
    """exclusive: an S2/S3 record belongs to at most one S1 (S1 is deduplicated) -> keep its best S1.
    rel: additionally require p >= rel * (best p of that S1 entity)."""
    p = np.asarray(p)
    keep = p >= threshold
    if exclusive:
        keep &= p >= pd.Series(p).groupby(ri).transform("max").values
    if rel > 0:
        keep &= p >= rel * pd.Series(p).groupby(li).transform("max").values
    return keep


def tune_decision(li, ri, p, label, n_true):
    """Grid-search the decision rule directly on macro F0.5 (the leaderboard metric)."""
    p = np.asarray(p)
    is_best_r = p >= pd.Series(p).groupby(ri).transform("max").values
    best_l = pd.Series(p).groupby(li).transform("max").values
    best = (-1, None)
    for t, ex, rel in itertools.product(np.round(np.arange(0.05, 0.96, 0.025), 3), [True, False],
                                        [0.0, 0.3, 0.6, 0.8]):
        keep = (p >= t) & (is_best_r if ex else True) & (p >= rel * best_l)
        s = per_entity_f05(li, keep, label, n_true).mean()
        if s > best[0]:
            best = (float(s), dict(threshold=float(t), exclusive=bool(ex), rel=float(rel)))
    return best


def oof_predict(X, y, groups, n_folds, rounds):
    """Out-of-fold probabilities; folds grouped by S1 entity so no entity leaks."""
    oof = np.zeros(len(y), np.float32)
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        b = lgb.train(LGB_PARAMS, lgb.Dataset(X.iloc[tr], label=y[tr]), num_boost_round=rounds)
        oof[va] = b.predict(X.iloc[va])
        print(f"  fold {k}: {len(va):,} pairs, pos={int(y[va].sum()):,} {mem()}")
        del b
        gc.collect()
    return oof


# %% [TRAIN]
def _right_sampler(r2s1, kept_s1, frac):
    """Keep all matches of the sampled S1 entities + the same fraction of unmatched S2/S3 records,
    so the sample looks like the full data (same match / distractor ratio)."""
    def keep(ids):
        h = _hash01(ids)
        return np.array([(r2s1[i] in kept_s1) if i in r2s1 else (hh < frac) for i, hh in zip(ids, h)])
    return keep


def train(cfg=None, frac=TRAIN_FRAC):
    cfg = cfg or BlockingConfig()
    t0 = time.time()
    os.makedirs(ARTIFACTS, exist_ok=True)
    r2s1, n_true_d = load_truth(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"))
    s1 = load_s1(DATA_DIR, "train", frac)
    kept_s1 = set(s1["entity_id"].tolist()) if frac < 1.0 else None
    print(f"train: {len(s1):,} S1 entities, {len(r2s1):,} true links in ground truth {mem()}")

    keep_right = _right_sampler(r2s1, kept_s1, frac) if frac < 1.0 else None

    cand, X, rid = run_blocking(DATA_DIR, "train", cfg, s1, keep_right)
    li, ri = cand["li"].values.astype(np.int64), cand["ri"].values.astype(np.int64)
    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    y = np.array([r2s1.get(r) == s for r, s in zip(rid, s1_ids[li])], np.int8)
    n_true = np.array([n_true_d.get(i, 0) for i in s1_ids])   # sampling keeps all matches of kept S1
    del r2s1
    gc.collect()

    f_oracle = per_entity_f05(li, y == 1, y, n_true).mean()
    print(f"blocking: pair recall = {y.sum() / max(n_true.sum(), 1):.4f}, "
          f"{len(cand) / len(s1):.2f} candidates per S1, oracle macro F0.5 = {f_oracle:.4f}")

    # bound RAM for LightGBM: sample S1 entities (all their pairs) if there are too many pairs
    ent_mask = np.ones(len(s1), bool)
    if len(cand) > MAX_TRAIN_PAIRS:
        ent_mask = _hash01(list(s1_ids)) < MAX_TRAIN_PAIRS / len(cand)
    m = ent_mask[li]
    Xm, ym, lim, rim = X[m].reset_index(drop=True), y[m], li[m], ri[m]
    remap = -np.ones(len(s1), np.int64)
    remap[ent_mask] = np.arange(ent_mask.sum())
    print(f"model data: {len(ym):,} pairs from {ent_mask.sum():,} S1 entities {mem()}")

    oof = oof_predict(Xm, ym, lim, N_FOLDS, ROUNDS)
    score, dec = tune_decision(remap[lim], rim, oof, ym, n_true[ent_mask])
    print(f"OOF macro F0.5 = {score:.4f} with {dec}")
    f_ent = per_entity_f05(remap[lim], decide(lim, rim, oof, **dec), ym, n_true[ent_mask])
    country = s1["country_n"].to_numpy(dtype=object)[ent_mask]
    for c in sorted(set(country)):
        print(f"  country={c or '<empty>'}: {f_ent[country == c].mean():.4f} ({(country == c).sum():,} S1)")

    print("fitting final model ...")
    booster = lgb.train(LGB_PARAMS, lgb.Dataset(Xm, label=ym), num_boost_round=ROUNDS)
    booster.save_model(os.path.join(ARTIFACTS, "model.txt"))
    imp = pd.Series(booster.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print("top features:\n" + imp.head(12).round(0).to_string())
    with open(os.path.join(ARTIFACTS, "config.json"), "w") as f:
        json.dump({"blocking": cfg.to_dict(), "decision": dec, "features": list(X.columns),
                   "oof_macro_f05": score}, f, indent=2)
    print(f"saved to {ARTIFACTS}  (total {time.time() - t0:.0f}s) {mem()}")


# %% [PREDICT + VALIDATE]
def write_lists(s1_ids, li, rid, col, path):
    df = pd.DataFrame({"li": li, "rid": rid}).drop_duplicates()
    lists = df.groupby("li")["rid"].agg(lambda x: ",".join(sorted(x)))
    out = pd.Series("", index=np.arange(len(s1_ids)), dtype=object)
    out[lists.index.values] = lists.values
    pd.DataFrame({"source1_entity_id": s1_ids, col: out.values}).to_csv(path, sep="\t", index=False)


def validate(split, match_path, cand_path):
    """Checks every submission rule without loading all S2/S3 rows into memory."""
    s1 = set(read_tsv(os.path.join(DATA_DIR, split, f"{split}_source1.tsv"), usecols=["entity_id"])["entity_id"])
    lists = {}
    for path, col in [(match_path, "matched_entity_ids"), (cand_path, "candidate_entity_ids")]:
        df = read_tsv(path)
        assert list(df.columns) == ["source1_entity_id", col], f"{path}: bad header"
        assert df["source1_entity_id"].is_unique, f"{path}: duplicate source1 rows"
        assert set(df["source1_entity_id"]) == s1, f"{path}: S1 ids missing or extra"
        m = {a: [x for x in b.split(",") if x] for a, b in zip(df["source1_entity_id"], df[col])}
        for k, v in m.items():
            assert len(v) == len(set(v)), f"{path}: duplicate ids in list of {k}"
        lists[col] = m
    used = set()
    for k, v in lists["matched_entity_ids"].items():
        assert set(v) <= set(lists["candidate_entity_ids"][k]), f"{k}: match not in candidate_pairs"
    for v in lists["candidate_entity_ids"].values():
        used.update(v)
    for src in (2, 3):   # every referenced id must exist in the S2/S3 files
        for ch in read_tsv(os.path.join(DATA_DIR, split, f"{split}_source{src}.tsv"),
                           chunksize=1_000_000, usecols=["entity_id"]):
            used.difference_update(ch["entity_id"].tolist())
    assert not used, f"{len(used)} ids are not S2/S3 ids of the {split} set, e.g. {list(used)[:3]}"
    print("validation OK: both files follow every submission rule")


def predict(split="test"):
    t0 = time.time()
    with open(os.path.join(ARTIFACTS, "config.json")) as f:
        conf = json.load(f)
    cfg = BlockingConfig(**conf["blocking"])
    booster = lgb.Booster(model_file=os.path.join(ARTIFACTS, "model.txt"))
    s1 = load_s1(DATA_DIR, split)
    print(f"{split}: {len(s1):,} S1 entities {mem()}")
    cand, X, rid = run_blocking(DATA_DIR, split, cfg, s1)
    li, ri = cand["li"].values.astype(np.int64), cand["ri"].values.astype(np.int64)
    assert list(X.columns) == conf["features"], "feature mismatch between train and predict"
    p = booster.predict(X)
    del X
    gc.collect()
    keep = decide(li, ri, p, **conf["decision"])

    os.makedirs(OUT_DIR, exist_ok=True)
    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    cand_path = os.path.join(OUT_DIR, "candidate_pairs.tsv")
    match_path = os.path.join(OUT_DIR, "matching_results.tsv")
    write_lists(s1_ids, li, rid, "candidate_entity_ids", cand_path)
    write_lists(s1_ids, li[keep], rid[keep], "matched_entity_ids", match_path)
    n_empty = len(s1) - len(np.unique(li[keep]))
    print(f"wrote {match_path}: {int(keep.sum()):,} links, {n_empty / len(s1):.1%} S1 with no match")
    print(f"wrote {cand_path}: {len(cand):,} pairs ({len(cand) / len(s1):.2f} per S1)  "
          f"({time.time() - t0:.0f}s)")
    validate(split, match_path, cand_path)


# %% [RUN]
if __name__ == "__main__":
    train()
    predict()
