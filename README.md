# Amazon ML Challenge: Business Entity Resolution

Links every Source 1 entity to its matching Source 2 and Source 3 records. Scored with macro F0.5, which favours precision.

## Pipeline

```
S1, S2, S3 ──► normalise ──► BLOCKING (TF-IDF kNN, both directions, per country)
                                   │   → candidate_pairs.tsv (exactly what the model scores)
                                   ▼
                             ~47 pairwise features (fuzzy name/address, postal, numbers, rank/gap)
                                   ▼
                             LightGBM (MIT)  → p(match)
                                   ▼
            decision rule tuned on out-of-fold macro F0.5:
            p ≥ t  AND  each S2/S3 record goes only to its best S1  (AND p ≥ rel·best-of-S1)
                                   ▼
                             matching_results.tsv
```

1. **Normalisation** (`er/text.py`): strip accents, map `&` to `and`, and collapse abbreviations to one canonical form (Corp/Corporation, Pvt/Private, St/Street, Bd/Boulevard, ordinals…). A *core name* drops legal suffixes (Inc, LLC, Pvt Ltd, SARL, SAS…). DBA/aka aliases are split out. Postal codes (US 5 digits, India 6 digits even when written "560 001", France 5 digits) and house numbers are extracted. There are no country-specific branches, so France works unchanged.
2. **Blocking** (`er/blocking.py`): each record becomes an L2-normalised sparse vector: char 2–4-grams of the core name plus address word tokens. Within each country block, exact kNN runs in three passes:
   * S1 → S2/S3, top `k_left`
   * S2/S3 → S1, top `k_right`. Source 1 is deduplicated, so each S2/S3 record belongs to at most one S1. Asking "which S1 is nearest to me?" catches S1s that have many duplicates while keeping lists short.
   * S1 → S2/S3 on the name alone, top `k_name` (catches missing or landmark-only addresses).

   The union is then pruned by absolute and relative similarity and capped per S1. The cost is O(N·k) with chunked sparse matmuls. At billions of records the same vectors go into an ANN index (FAISS/HNSW) per country block.
3. **Features** (`er/features.py`): RapidFuzz ratio, token_set, token_sort, partial, Jaro-Winkler and Levenshtein on names and addresses; TF-IDF cosines; alias best-match; acronym match; postal equal/different/missing; house-number match; and *competition features* (how far this pair is below the best candidate for the same S1 and for the same S2/S3 record). The country value itself is never used as a feature, only whether the two records agree on it.
4. **Model** (`er/model.py`): LightGBM binary classifier. Out-of-fold predictions come from GroupKFold grouped by S1 entity.
5. **Decision rule**: grid-searched directly on macro F0.5 over the OOF predictions (threshold, exclusive assignment, relative-to-best). Recall counts true links that blocking missed, so the score is honest.

## Usage

```bash
pip install -r requirements.txt
# data goes in dataset/train/*.tsv and dataset/test/*.tsv

python tune_blocking.py --data dataset --min_recall 0.98   # choose the smallest candidate config
python train.py   --data dataset --k_left 3 --k_right 1 --k_name 1   # prints OOF F0.5, saves artifacts/
python predict.py --data dataset                            # writes output/*.tsv and validates them
```

`train.py` prints the blocking report (pair recall, candidates per S1, reduction ratio, oracle F0.5) and the OOF macro F0.5 per country. `predict.py` runs `validate_submission.py`, which checks every submission rule: one row per S1, only existing S2/S3 ids, no duplicates, matches ⊆ candidates.

Smoke test without the real data:

```bash
python tools/make_synthetic.py --out synthetic
python train.py --data synthetic && python predict.py --data synthetic
```

## Licences
Everything is permissive: LightGBM (MIT), scikit-learn (BSD-3), RapidFuzz (MIT), pandas/numpy/scipy (BSD). There is no model over 8B parameters.
