# xleap_fel_research

## Fetching archived PV snapshots

The `xleap_parser.snapshots` module fetches binned PV values from the MEME
archiver over a UTC time window and writes a deterministic CSV
(`nominal_time, pv, timestamp, value, moved, spread`).

Run from `xleap_parser/` so the default `pvs_sxr.csv` resolves:

```bash
cd xleap_parser

python -m snapshots \
  --from "2026-06-01T00:00:00.000Z" \
  --to   "2026-07-01T00:00:00.000Z" \
  --pvs  pvs_sxr.csv \
  --out  snapshots_2026-06.csv \
  --jobs 16
```

The window is half-open: `[from, to)`. The example above pulls all of
June 2026 by ending at `2026-07-01T00:00:00Z`.

### Common flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--from` / `--to` | Jan 2018 window (see `snapshots/config.py`) | ISO8601 UTC window bounds |
| `--pvs` | `pvs_sxr.csv` | Input CSV of PV names (one per row, `PV` header optional) |
| `--out` | `snapshots.csv` | Output CSV path |
| `--jobs` | `16` (or `$JOBS`) | Concurrent archiver requests |
| `--bin-seconds` | `900` | Nominal-grid bin width in seconds |
| `--operator` | `firstSample` | Bin-reduction fn; `""` for raw samples |
| `--protocol` | `PVA` | `PVA` (EPICS v4) or `HTTP` (appliance) |
| `--config` | none | Optional TOML file of overrides (CLI flags win) |

Config precedence, low → high: dataclass defaults → `--config` TOML → CLI flags.

### Python API

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from snapshots import RunConfig, fetch_pv
from snapshots.__main__ import read_pvs

cfg = RunConfig(
    from_time="2026-06-01T00:00:00.000Z",
    to_time="2026-07-01T00:00:00.000Z",
)
pvs = read_pvs("pvs_sxr.csv")
with ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(fetch_pv, pv, cfg) for pv in pvs]
    rows = [r for f in as_completed(futures) for r in f.result()[1]]
```

### Sanity check

For a month at the default 15-min bin: 30 × 96 ≈ 2,880 rows per PV. With
~27 PVs in `pvs_sxr.csv`, expect ~77k rows / ~8 MB. Errors are printed to
stderr per-PV; the final `Done. rows=… empty_pvs=… errors=…` line
summarizes the run.

If a short window shows transient PVA errors on the first attempt, retry
— cold-start auth handshakes can drop a few requests before the
connection warms up.
