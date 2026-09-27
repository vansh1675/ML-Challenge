"""Text normalisation for business names and addresses.

Everything here is country-agnostic on purpose: the test set contains a country
(France) that never appears in training, so we only use generic rules plus a few
multilingual abbreviation maps.
"""
import re
import unicodedata

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
