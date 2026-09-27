# Amazon ML Challenge: Business Entity Resolution

Links every Source 1 entity to its matching Source 2 and Source 3 records. Scored with macro F0.5, which favours precision.

## Pipeline

```
S1, S2, S3 ──► normalise (names, addresses, states, city renames)
                 │
                 ▼
  RETRIEVAL pool (wide, cheap):  FAISS HNSW on SVD(TF-IDF name char-3grams + address words)
                                 per country, S1→S2/S3 and S2/S3→S1
                               + exact-key hash joins (core name, sorted name tokens, house#+street)
                 ▼
  SELECTION (exact cosine):      top-k per S1, top-k per S2/S3 record, similarity floors, per-S1 cap
                 │   → candidate_pairs.tsv  (exactly the pairs the model scores; ~1.8–2.7 per S1)
                 ▼
  ~49 pairwise features → LightGBM (MIT) → p(match)
                 ▼
  decision rule tuned on out-of-fold macro F0.5:
  p ≥ t,  each S2/S3 record goes only to its best S1,  p ≥ rel·(best p of that S1)
                 ▼
  matching_results.tsv
```

1. **Normalisation** (`er/text.py`):
   * strips accents; maps `&` to `and`; handles `L.L.C.`/`P.C.` and possessives
   * collapses abbreviation variants (Corp/Corporation, Pvt/Private, Rd/Road, Bd/Boulevard, Nagar/Ngr, 1st/1…)
   * maps US and Indian state names to codes (Texas→tx, Uttar Pradesh→up, West Bengal→wb)
   * maps old city names to current ones (Calcutta→kolkata, Bombay→mumbai, Bangalore→bengaluru…)
   * builds a *core name* without legal suffixes (LLC, Pvt Ltd, SARL…)
   * extracts the postal code, a "house number + street word" key, and the last address component (region)

   None of these are country-specific code paths, so the unseen France set goes through the same code.
2. **Blocking** (`er/blocking.py`) is built to scale:
   * **Retrieval** makes a wide pool of plausible pairs (about 18 per S1). Each record's sparse TF-IDF vector is reduced with TruncatedSVD (fitted on a sample) to a 128-d dense vector. A FAISS HNSW index per country is queried both ways: S1 → S2/S3 (top 10) and S2/S3 → S1 (top 3). Source 1 is deduplicated, so each S2/S3 record belongs to at most one S1, and the reverse query recovers S1 entities with many duplicates. Exact-key hash joins run alongside it with a bucket-size cap.
   * **Selection** computes exact sparse cosines on the pool. A pair is kept if it is in the top `k_left` for its S1, the top `k_right` for its S2/S3 record, or the top `k_name` by name. It must pass similarity floors, and each S1 is capped. Pairs that are some record's top choice are kept first when the cap applies.
   * Cost is O(N log N). At billions of records, FAISS indexes shard per country or region and the key joins become map-reduce group-bys.
3. **Features** (`er/features.py`, computed in chunks):
   * RapidFuzz name and address similarities; TF-IDF cosines
   * alias (DBA) and acronym matches
   * whether postal code, street key and region are equal, different or missing
   * house-number overlap
   * *competition features*: how far this pair is below the best rival for the same S1 and for the same S2/S3 record

   The country value itself is never a feature, only whether the two records agree on it.
4. **Model** (`er/model.py`): LightGBM binary classifier. Out-of-fold predictions come from GroupKFold grouped by S1 entity.
5. **Decision rule**: grid-searched directly on macro F0.5 over the OOF predictions. Recall counts true links that blocking missed, so the reported score is honest.

## Usage

```bash
pip install -r requirements.txt
# data goes in dataset/train/*.tsv and dataset/test/*.tsv

# 1) choose the smallest candidate set that keeps recall (prints recall / candidates-per-S1 / oracle F0.5)
python tune_blocking.py --data dataset --min_recall 0.97

# 2) train + tune the decision rule (pass the chosen selection flags; --sample_s1 speeds up big data)
python train.py --data dataset --k_left 2 --k_right 1 --k_name 0 --max_per_s1 5 --sample_s1 300000 --folds 3

# 3) test predictions -> output/matching_results.tsv + output/candidate_pairs.tsv (validated)
python predict.py --data dataset
```

`train.py` prints the blocking report (pair recall, candidates per S1, reduction ratio, oracle F0.5) and the OOF macro F0.5 per country. `predict.py` runs `validate_submission.py`, which checks every submission rule: one row per S1, only existing S2/S3 ids, no duplicates, matches ⊆ candidates.

Smoke test without the real data:

```bash
python tools/make_synthetic.py --out synthetic
python train.py --data synthetic && python predict.py --data synthetic
```

## Licences
Everything is permissive: LightGBM (MIT), FAISS (MIT), scikit-learn (BSD-3), RapidFuzz (MIT), pandas/numpy/scipy (BSD). There is no model over 8B parameters.

## Low-memory streaming version (`entity_resolution.py`)

For the real data size (about 1.6M S1 and 10M S2/S3 rows) on an 8 GB laptop, use the single-file
`entity_resolution.py`:

* Source 1 is indexed once: TF-IDF fitted on a 300k sample, SVD to 64 dimensions, one FAISS HNSW index
  per country, and exact-key tables stored as uint64 hashes.
* S2/S3 are streamed in chunks of 200k rows. Each chunk is normalised, cached to disk, and matched
  against the S1 index. The candidate pool is pruned after every chunk.
* Pass 2 re-reads the cached chunks and computes features only for the final candidates, written
  into one preallocated float32 matrix.
* `TRAIN_FRAC` trains on a cluster-preserving sample of S1 entities (each sampled entity keeps all its
  matches, and the same fraction of unmatched S2/S3 records is kept).

Measured on synthetic data of the real size (1.6M S1 + 10.4M S2/S3, 4 CPUs): training with
`TRAIN_FRAC=1.0` took 28 minutes with a 5.1 GB peak, and that was before the pass-2 preallocation
removed about 1.2 GB. With `TRAIN_FRAC=0.5` the peak is about half.
