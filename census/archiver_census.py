"""Tier B census: DELIVERED 1 Hz archiver rows per (mode x machine).

Tier A counts *booked* shifts. This counts the *real* thing the mentor asked for
-- how many archiver data points actually exist per cell -- by joining every
archiver row's timestamp against the scheduled calendar events (the identical
interval-join Tier A uses, from :mod:`census.common`) and, independently,
gating on delivery (the FEE gas-detector target above a noise floor).

Per cell it reports three counts, so "scheduled but never lased" is visible
rather than hidden:

* ``n_total``     -- archiver rows whose timestamp falls in a shift of that mode.
* ``n_delivered`` -- of those, rows with ``GDET:FEE1:241:ENRC`` above the floor
  (beam actually lasing -- the honest denominator for a training set).
* ``n_unscheduled`` (mode ``__unscheduled__``) -- rows in no scheduled shift.

**Scope.** Needs pandas + the S3DF ``.pkl`` set, so it runs on ``iana`` only
(``/sdf/data`` is not mounted on the ``sdfssh`` login nodes). The dataset is
**HXR-only** today; SXR cells stay empty until an SXR archiver pull lands, which
is itself a headline result -- the numbers show *why* SXR must be pulled before
any XLEAP-SXR model.

Run (on ``iana``, in the ``emittance`` env)::

    ~/.conda/envs/emittance/bin/python -m census.archiver_census \\
        --dataset-dir /sdf/data/ad/ard/u/zihanzhu/ml/lcls_fel_tuning/dataset_updated

Delivery gate ``--noise-floor`` is provisional (mJ); confirm the true FEE floor
with David Cesar / Zihan before trusting ``n_delivered`` as a training count.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .common import Event, assign_event, load_events

# Default S3DF dataset + the target/gate column (see project CLAUDE.md 3.5).
_DEFAULT_DATASET = Path("/sdf/data/ad/ard/u/zihanzhu/ml/lcls_fel_tuning/dataset_updated")
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CAL = _REPO_ROOT.parent / "data_samples" / "program_calendar_data" / "cleaned"
_DEFAULT_OUT = _REPO_ROOT.parent / "data_samples" / "census" / "archiver_census_hxr.csv"

GDET_TARGET = "GDET:FEE1:241:ENRC"
UNSCHEDULED = "__unscheduled__"


@dataclass
class RowCounts:
    """Delivered/total row tallies for one ``(mode, machine)`` cell."""

    n_total: int = 0
    n_delivered: int = 0


def tally_rows(
    epochs: list[float],
    delivered: list[bool],
    events: list[Event],
) -> dict[tuple[str, str], RowCounts]:
    """Assign each row to its scheduled cell and tally total/delivered counts.

    Pure (no pandas): ``epochs`` are POSIX seconds, ``delivered[i]`` is whether
    row ``i`` cleared the delivery gate. Rows in no shift land in the
    ``(UNSCHEDULED, "unknown")`` cell. Kept pandas-free so the join logic is
    unit-tested on this laptop, identical to what runs on the cluster.
    """
    cells: dict[tuple[str, str], RowCounts] = defaultdict(RowCounts)
    for epoch, is_delivered in zip(epochs, delivered):
        event = assign_event(events, epoch)
        key = (event.mode or "unlabeled", event.machine) if event else (UNSCHEDULED, "unknown")
        cell = cells[key]
        cell.n_total += 1
        if is_delivered:
            cell.n_delivered += 1
    return cells


def _month_epochs_delivered(path, gdet_col: str, floor: float):
    """Read one archiver ``.pkl`` -> (epochs[s], delivered[bool]). pandas-local."""
    import pandas as pd  # local import: only Tier B needs pandas / the cluster

    frame = pd.read_pickle(path)
    index = frame.index
    if index.tz is None:
        raise ValueError(f"{path.name}: index is tz-naive; expected tz-aware US/Pacific")
    # tz-aware -> int64 ns since UNIX epoch (UTC) regardless of the stored zone.
    epochs = (index.tz_convert("UTC").astype("int64") / 1e9).tolist()
    if gdet_col not in frame.columns:
        raise ValueError(f"{path.name}: missing target column {gdet_col!r}")
    delivered = (pd.to_numeric(frame[gdet_col], errors="coerce") > floor).fillna(False)
    return epochs, delivered.tolist()


def run(
    dataset_dir: Path,
    calendar_csv: Path,
    line: str,
    glob: str,
    gdet_col: str,
    floor: float,
) -> tuple[dict[tuple[str, str], RowCounts], list[str]]:
    """Tally every matching monthly ``.pkl`` against ``line``'s calendar events."""
    events = load_events(calendar_csv, line)
    totals: dict[tuple[str, str], RowCounts] = defaultdict(RowCounts)
    logs: list[str] = [f"loaded {len(events)} {line} calendar events from {calendar_csv.name}"]

    months = sorted(dataset_dir.glob(glob))
    if not months:
        logs.append(f"! no files match {glob} in {dataset_dir}")
        return totals, logs

    for path in months:
        try:
            epochs, delivered = _month_epochs_delivered(path, gdet_col, floor)
        except Exception as exc:  # noqa: BLE001 - report and skip a bad month, don't abort
            logs.append(f"! {path.name}: {exc}")
            continue
        cells = tally_rows(epochs, delivered, events)
        for key, cell in cells.items():
            totals[key].n_total += cell.n_total
            totals[key].n_delivered += cell.n_delivered
        logs.append(f"  {path.name}: {len(epochs):>7,} rows")
    return totals, logs


def write_csv(totals: dict[tuple[str, str], RowCounts], line: str, out_path: Path) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["line", "mode", "machine", "n_total", "n_delivered", "delivered_frac"])
        for (mode, machine), cell in sorted(totals.items()):
            frac = cell.n_delivered / cell.n_total if cell.n_total else 0.0
            writer.writerow([line, mode, machine, cell.n_total, cell.n_delivered, round(frac, 4)])


def _print_report(totals: dict[tuple[str, str], RowCounts], line: str, logs: list[str]) -> None:
    print("\n".join(logs))
    print("\n" + "=" * 78)
    print(f"TIER B -- DELIVERED archiver-row census  (line: {line})")
    print("=" * 78)
    print(f"\n{'mode':<16}{'machine':<9}{'n_total':>12}{'n_delivered':>13}{'deliv %':>9}")
    print("-" * 59)
    for (mode, machine), cell in sorted(totals.items()):
        frac = 100 * cell.n_delivered / cell.n_total if cell.n_total else 0.0
        print(f"{mode:<16}{machine:<9}{cell.n_total:>12,}{cell.n_delivered:>13,}{frac:>8.1f}%")
    print(
        "\nn_delivered gates on "
        f"{GDET_TARGET} > floor -- confirm the true floor with David Cesar/Zihan."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--calendar-csv", type=Path, default=None,
                        help="classified calendar CSV (default: cleaned/<line>_classified_all.csv)")
    parser.add_argument("--line", default="HXR", choices=["HXR", "SXR"])
    parser.add_argument("--glob", default=None, help="pkl glob (default: <line>_archiver_*.pkl)")
    parser.add_argument("--gdet-col", default=GDET_TARGET)
    parser.add_argument("--noise-floor", type=float, default=0.05,
                        help="FEE mJ above which a row counts as delivered (provisional)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    calendar_csv = args.calendar_csv or (_DEFAULT_CAL / f"{args.line.lower()}_classified_all.csv")
    glob = args.glob or f"{args.line.lower()}_archiver_*.pkl"
    out = args.out or (_DEFAULT_OUT.parent / f"archiver_census_{args.line.lower()}.csv")

    if not args.dataset_dir.exists():
        print(f"error: dataset dir not found: {args.dataset_dir}\n"
              "       Tier B runs on `iana` only (/sdf/data is not mounted on login nodes).",
              file=sys.stderr)
        return 2
    if not calendar_csv.exists():
        print(f"error: calendar CSV not found: {calendar_csv}", file=sys.stderr)
        return 2

    totals, logs = run(args.dataset_dir, calendar_csv, args.line, glob, args.gdet_col, args.noise_floor)
    _print_report(totals, args.line, logs)
    if totals:
        write_csv(totals, args.line, out)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
