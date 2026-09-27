"""Generates a small noisy dataset in the challenge layout, for smoke-testing the pipeline only.

    python tools/make_synthetic.py --out synthetic
"""
import argparse
import os
import random

import pandas as pd

WORDS = ["apex", "blue", "river", "summit", "golden", "star", "metro", "green", "silver", "royal", "prime",
         "sunrise", "lotus", "eagle", "oak", "harbor", "crystal", "ganesh", "shree", "lakshmi", "krishna",
         "pacific", "atlas", "nova", "zenith", "maple", "cedar", "delta", "orion", "vertex", "bharat", "sagar",
         "bleu", "soleil", "lumiere", "jardin", "etoile"]
KINDS = ["Traders", "Technologies", "Solutions", "Enterprises", "Foods", "Motors", "Pharma", "Textiles",
         "Consulting", "Logistics", "Builders", "Electronics", "Bakery", "Boulangerie", "Services"]
SUFFIX = {"US": ["Inc", "LLC", "Corp", "Corporation", "Co"], "India": ["Pvt Ltd", "Private Limited", "Ltd", "LLP"],
          "France": ["SARL", "SAS", "SA", "EURL"]}
STREETS = {"US": ["Main Street", "Oak Avenue", "Park Road", "Sunset Boulevard", "Lake Drive"],
           "India": ["MG Road", "Station Road", "Nehru Nagar", "Gandhi Marg", "Sector 14"],
           "France": ["Rue de la Paix", "Avenue Victor Hugo", "Boulevard Saint-Germain", "Rue Lafayette"]}
CITIES = {"US": ["Austin, TX", "Denver, CO", "Seattle, WA"], "India": ["Pune, Maharashtra", "Bengaluru, Karnataka"],
          "France": ["Paris", "Lyon", "Marseille"]}
ABBR = {"Street": "St", "Avenue": "Ave", "Road": "Rd", "Boulevard": "Blvd", "Drive": "Dr", "Private": "Pvt",
        "Limited": "Ltd", "Corporation": "Corp", "Technologies": "Tech", "Services": "Svcs", "Saint": "St"}


def typo(s, r):
    if len(s) < 4 or r.random() > 0.3:
        return s
    i = r.randrange(1, len(s) - 1)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def noisy(name, addr, r):
    for k, v in ABBR.items():
        if r.random() < 0.5:
            name, addr = name.replace(k, v), addr.replace(k, v)
    if r.random() < 0.3:
        name = name.rsplit(" ", 1)[0]
    if r.random() < 0.2:
        name = name.upper()
    name = typo(name, r).replace(" and ", " & ")
    parts = addr.split(", ")
    if r.random() < 0.3 and len(parts) > 2:
        parts = parts[:-1]
    if r.random() < 0.2:
        parts = [parts[0], "Near SBI ATM"] + parts[1:]
    return name, typo(", ".join(parts), r)


def make(split, n, countries, r, out, start):
    s1, s2, s3, gt = [], [], [], []
    c2 = c3 = 0
    for i in range(n):
        c = r.choice(countries)
        name = f"{r.choice(WORDS).title()} {r.choice(WORDS).title()} {r.choice(KINDS)} {r.choice(SUFFIX[c])}"
        pin = str(r.randint(10000, 99999)) if c != "India" else str(r.randint(100000, 999999))
        addr = f"{r.randint(1, 999)} {r.choice(STREETS[c])}, {r.choice(CITIES[c])}, {pin}"
        sid = f"S1-{start + i:05d}"
        s1.append((sid, name, addr, c))
        m = []
        for _ in range(r.choice([0, 0, 1, 1, 1, 2, 3])):
            nn, aa = noisy(name, addr, r)
            if r.random() < 0.5:
                c2 += 1; eid = f"S2-{start + c2:05d}"; s2.append((eid, nn, aa, c))
            else:
                c3 += 1; eid = f"S3-{start + c3:05d}"; s3.append((eid, nn, aa, c))
            m.append(eid)
        gt.append((sid, ",".join(m)))
    for _ in range(n // 3):  # unmatched distractors
        c = r.choice(countries)
        nn = f"{r.choice(WORDS).title()} {r.choice(WORDS).title()} {r.choice(KINDS)}"
        aa = f"{r.randint(1, 999)} {r.choice(STREETS[c])}, {r.choice(CITIES[c])}"
        c2 += 1; s2.append((f"S2-{start + c2:05d}", nn, aa, c))
    d = os.path.join(out, split)
    os.makedirs(d, exist_ok=True)
    cols = ["entity_id", "business_name", "business_address", "country"]
    for nm, rows in [("source1", s1), ("source2", s2), ("source3", s3)]:
        pd.DataFrame(rows, columns=cols).sample(frac=1, random_state=1).to_csv(
            os.path.join(d, f"{split}_{nm}.tsv"), sep="\t", index=False)
    if split == "train":
        pd.DataFrame(gt, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
            os.path.join(d, "train_ground_truth.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    r = random.Random(0)
    make("train", a.n, ["US", "India"], r, a.out, 0)
    make("test", a.n // 2, ["US", "India", "France"], r, a.out, 50000)
