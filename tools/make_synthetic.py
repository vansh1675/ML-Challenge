"""Generates a noisy dataset in the challenge layout (formats mimic the real sample),
for smoke-testing / timing the pipeline only. Real scores will be lower.

    python tools/make_synthetic.py --out synthetic --n 20000
"""
import argparse
import os
import random

import pandas as pd

FIRST = ["Stacia", "Asset", "Pryor", "Vedas", "Optimal", "Tech", "Perfect", "Delvalle", "Harris", "Gonzalez",
         "Summit", "Blue", "River", "Golden", "Metro", "Royal", "Prime", "Sunrise", "Lotus", "Eagle", "Harbor",
         "Shree", "Ganesh", "Lakshmi", "Krishna", "Bharat", "Sagar", "Apex", "Atlas", "Nova", "Zenith", "Maple",
         "Cedar", "Orion", "Vertex", "Laycock", "Mirza", "Elspeth", "Patriot", "Kogyo", "Good", "DX", "Star"]
KIND = ["Supply", "Building Foundation", "Marketing", "Exports", "Communications", "Food", "Exchange",
        "Technologies", "Solutions", "Enterprises", "Motors", "Pharma", "Textiles", "Consulting", "Logistics",
        "Builders", "Electronics", "Services", "Traders", "Industries", "Associates"]
SUFFIX = {"US": ["LLC", "L.L.C.", "Inc", "Corp", "Corporation", "P.C.", "Co", ""],
          "India": ["Pvt Ltd", "Private Limited", "Ltd", "LLP", ""],
          "France": ["SARL", "SAS", "SA", "EURL", ""]}
US_STREETS = ["County Road", "Lowell Road", "Cliff Lodge Drive", "Victorian Lane", "Roundwood Place",
              "Lisburn Terrace Lane", "Main Street", "Oak Avenue", "Sunset Boulevard"]
US_CITIES = [("Hughes Springs", "TX", "Texas"), ("Concord", "MA", "Massachusetts"), ("Sandy", "UT", "Utah"),
             ("Brookville", "NY", "New York"), ("Tucson", "AZ", "Arizona"), ("Houston", "TX", "Texas")]
IN_AREAS = ["Gomti Nagar", "Vishwas Khand", "Shivalika", "Gole Bldg", "Khopat", "Sector 14", "MG Road"]
IN_CITIES = [("Kolkata", "Calcutta", "West Bengal"), ("Mumbai", "Bombay", "Maharashtra"),
             ("Lucknow", "Lucknow", "Uttar Pradesh"), ("Bengaluru", "Bangalore", "Karnataka"),
             ("Thane", "Thane", "Maharashtra"), ("Chennai", "Madras", "Tamil Nadu")]
FR_STREETS = ["Rue de la Paix", "Avenue Victor Hugo", "Boulevard Saint-Germain", "Rue Lafayette"]
FR_CITIES = ["Paris", "Lyon", "Marseille", "Toulouse"]
ABBR = [("Road", "Rd"), ("Street", "St"), ("Drive", "Dr"), ("Lane", "Ln"), ("Avenue", "Ave"),
        ("Boulevard", "Bd"), ("Private", "Pvt"), ("Limited", "Ltd"), ("Corporation", "Corp"),
        ("Technologies", "Tech"), ("Nagar", "Ngr"), (" and ", " & ")]


def typo(s, r, p=0.3):
    if len(s) < 5 or r.random() > p:
        return s
    i = r.randrange(1, len(s) - 2)
    op = r.random()
    if op < 0.4:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op < 0.7:
        return s[:i] + s[i + 1:]
    return s[:i] + s[i] + s[i:]


def make_record(c, r):
    name = f"{r.choice(FIRST)} {r.choice(KIND)}"
    if r.random() < 0.3:
        name = f"{r.choice(FIRST)}, {r.choice(FIRST)} and {r.choice(FIRST)} {r.choice(KIND)}"
    name = (name + " " + r.choice(SUFFIX[c])).strip()
    if c == "US":
        city, st, full = r.choice(US_CITIES)
        comps = dict(num=str(r.randint(1, 9999)), street=r.choice(US_STREETS), city=city, st=st, full=full,
                     unit=f"Unit {r.choice('ABH')}{r.randint(1, 999)}" if r.random() < 0.2 else "")
    elif c == "India":
        city, old, state = r.choice(IN_CITIES)
        comps = dict(num=f"{r.randint(1, 9)}/{r.randint(1, 999)}", area=r.choice(IN_AREAS), city=city, old=old,
                     state=state, pin=str(r.randint(110000, 799999)), shop=f"Shop No.{r.randint(1, 99)}")
    else:
        comps = dict(num=str(r.randint(1, 200)), street=r.choice(FR_STREETS), city=r.choice(FR_CITIES),
                     cp=str(r.randint(10000, 95999)))
    return name, comps


def render(c, comps, r, noisy):
    if c == "US":
        st = comps["full"] if noisy and r.random() < 0.2 else comps["st"]
        parts = [f"{comps['num']} {comps['street']}"] + ([comps["unit"]] if comps["unit"] else []) + [comps["city"], st]
        if noisy and r.random() < 0.2:
            parts = [parts[-2]] + parts[:-2] + [parts[-1]]      # "Sandy, 9320 Cliff ..., UT"
        if noisy and r.random() < 0.15:
            parts = [p for p in parts if p != comps["city"]]
    elif c == "India":
        city = comps["old"] if noisy and r.random() < 0.3 else comps["city"]
        parts = [comps["shop"] if r.random() < 0.5 else comps["num"], comps["area"], city,
                 f"{comps['state']} {comps['pin']}"]
        if noisy and r.random() < 0.3:
            parts[-1] = comps["state"]                           # missing PIN
        if noisy and r.random() < 0.2:
            parts.insert(1, "Near SBI ATM")
    else:
        parts = [f"{comps['num']} {comps['street']}", f"{comps['cp']} {comps['city']}"]
        if noisy and r.random() < 0.3:
            parts = [f"{comps['num']} {comps['street']}", comps["city"]]
    s = ", ".join(parts)
    if noisy:
        for a, b in ABBR:
            if r.random() < 0.5:
                s = s.replace(a, b)
        s = typo(s, r, 0.2)
    return s


def noisy_name(name, r):
    for a, b in ABBR:
        if r.random() < 0.5:
            name = name.replace(a, b)
    if r.random() < 0.25:
        name = name.rsplit(" ", 1)[0]        # drop legal suffix
    if r.random() < 0.1:
        name = name.upper()
    if r.random() < 0.1:
        w = name.split(" ")
        if len(w) > 2:
            w[0], w[1] = w[1], w[0]
            name = " ".join(w)
    return typo(name, r, 0.25)


_used = set()


def uid(prefix, r):
    while True:
        x = f"{prefix}-{r.randint(10 ** 8, 10 ** 9 - 1)}"
        if x not in _used:
            _used.add(x)
            return x


def make(split, n, countries, r, out, n_distract):
    s1, rr, gt = [], [], []
    for i in range(n):
        c = r.choice(countries)
        name, comps = make_record(c, r)
        sid = uid("S1", r)
        s1.append((sid, name, render(c, comps, r, False), c))
        m = []
        for _ in range(r.choice([0, 0, 0, 1, 1, 1, 1, 2, 3])):
            src = r.choice(["S2", "S3"])
            eid = uid(src, r)
            rr.append((eid, noisy_name(name, r), render(c, comps, r, True), c))
            m.append(eid)
        gt.append((sid, ",".join(m)))
    for _ in range(n_distract):   # unmatched records, some sharing name or address with an S1
        c = r.choice(countries)
        name, comps = make_record(c, r)
        rr.append((uid(r.choice(["S2", "S3"]), r), noisy_name(name, r),
                   render(c, comps, r, True), c))
    d = os.path.join(out, split)
    os.makedirs(d, exist_ok=True)
    cols = ["entity_id", "business_name", "business_address", "country"]
    pd.DataFrame(s1, columns=cols).to_csv(os.path.join(d, f"{split}_source1.tsv"), sep="\t", index=False)
    R = pd.DataFrame(rr, columns=cols).drop_duplicates("entity_id").sample(frac=1, random_state=0)
    R[R["entity_id"].str.startswith("S2")].to_csv(os.path.join(d, f"{split}_source2.tsv"), sep="\t", index=False)
    R[R["entity_id"].str.startswith("S3")].to_csv(os.path.join(d, f"{split}_source3.tsv"), sep="\t", index=False)
    if split == "train":
        pd.DataFrame(gt, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
            os.path.join(d, "train_ground_truth.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    r = random.Random(0)
    make("train", a.n, ["US", "India"], r, a.out, a.n // 2)
    make("test", a.n // 2, ["US", "India", "France"], r, a.out, a.n // 4)
