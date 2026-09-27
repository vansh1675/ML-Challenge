"""Pairwise features for (S1 record, S2/S3 record) candidate pairs.

No feature uses the country value itself (the test set contains an unseen
country), only whether the two records agree on it.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein


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


def build_features(cand: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, vecs) -> pd.DataFrame:
    li, ri = cand["li"].values, cand["ri"].values
    L, R = left.iloc[li], right.iloc[ri]
    ln, rn = L["name_core"].tolist(), R["name_core"].tolist()
    lf, rf = L["name_full"].tolist(), R["name_full"].tolist()
    la, ra = L["addr"].tolist(), R["addr"].tolist()

    F = pd.DataFrame(index=cand.index)
    # --- blocking similarities ------------------------------------------------
    F["sim_comb"] = cand["sim_comb"].values
    F["sim_name"] = cand["sim_name"].values
    F["cos_name_word"] = vecs.pair_cosine("name_word", li, ri)
    F["cos_addr_word"] = vecs.pair_cosine("addr_word", li, ri)
    F["cos_addr_char"] = vecs.pair_cosine("addr_char", li, ri)

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
    jac, first = _num_feats(L["addr_nums"].tolist(), R["addr_nums"].tolist())
    F["a_num_jacc"] = jac
    F["a_num_first"] = first
    al = np.array([len(x.split()) for x in la], np.float32)
    ar = np.array([len(x.split()) for x in ra], np.float32)
    F["a_len_ratio"] = np.minimum(al, ar) / np.maximum(np.maximum(al, ar), 1)

    # --- context ------------------------------------------------------------------
    F["country_eq"] = _tristate(L["country_n"].values, R["country_n"].values)
    F["is_s3"] = (R["source"].values == "S3").astype(np.int8)

    # --- relative / competition features (how does this pair rank among rivals?) --
    g_l = cand["li"].values
    g_r = cand["ri"].values
    for col in ["sim_comb", "sim_name", "n_tset", "a_tset"]:
        v = pd.Series(F[col].values)
        F[f"{col}_gap_l"] = (v.groupby(g_l).transform("max") - v).values
        F[f"{col}_gap_r"] = (v.groupby(g_r).transform("max") - v).values
    F["rank_l"] = pd.Series(F["sim_comb"].values).groupby(g_l).rank(ascending=False, method="min").values
    F["rank_r"] = pd.Series(F["sim_comb"].values).groupby(g_r).rank(ascending=False, method="min").values
    F["ncand_l"] = pd.Series(g_l).map(pd.Series(g_l).value_counts()).values
    F["ncand_r"] = pd.Series(g_r).map(pd.Series(g_r).value_counts()).values
    return F.astype(np.float32)
