# Data census — how much data per `(mode × line × machine)` cell

Answers "which configurations have the most data" (XLEAP-HXR-NC = N1, XLEAP-SXR-NC
= N2, …) so model-cell prioritization is driven by numbers, not assumption.

Two tiers share one interval-join + NC/SC rule (`common.py`), so "which mode was
running at time *t*" is defined once.

| Tier | Unit | Deps | Where it runs | Output |
|------|------|------|---------------|--------|
| A `calendar_census` | **scheduled** shift-hours & events | stdlib | anywhere | `data_samples/census/calendar_census.csv` |
| B `archiver_census` | **delivered** 1 Hz archiver rows | pandas + S3DF | `iana` only | `data_samples/census/archiver_census_<line>.csv` |

## Tier A — runs now

```bash
cd ~/xleap_project/xleap_fel_research
python3 -m census.calendar_census          # prints pivots + writes long CSV
python3 -m census.calendar_census --no-write
```

Counts **booked shifts**, not delivered beam — a scheduled XLEAP shift that never
lased still counts. `primary_class` cells partition the total; the `all_flags`
table overlaps (a shift can wear several mode flags).

## Tier B — runs on the cluster

Needs pandas and the S3DF `.pkl` set (`/sdf/data` is mounted on `iana` nodes, not
the `sdfssh` login nodes). Joins every archiver row against the calendar events
and, independently, gates on the FEE target to separate *delivered* from merely
*scheduled*.

```bash
ssh iana
cd ~/xleap_project/xleap_fel_research
~/.conda/envs/emittance/bin/python -m census.archiver_census \
    --dataset-dir /sdf/data/ad/ard/u/zihanzhu/ml/lcls_fel_tuning/dataset_updated \
    --line HXR --noise-floor 0.05
```

Per cell it reports `n_total` (rows in a shift of that mode), `n_delivered`
(subset above the FEE floor — the honest training-set size), and an
`__unscheduled__` bucket (rows in no shift).

**Scope:** the dataset is **HXR-only** today, so SXR cells stay empty — which is
itself the result: it quantifies *why* an SXR archiver pull must precede any
XLEAP-SXR model. Rerun with `--line SXR` once that pull exists.

**Confirm before trusting `n_delivered`:** `--noise-floor` (mJ, default 0.05) is
provisional — get the true FEE floor from David Cesar / Zihan.

## Tests

```bash
python3 -m pytest census/tests/test_census.py -q   # pure join logic, no cluster
```
