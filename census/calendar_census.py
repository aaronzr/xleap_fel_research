"""Tier A census: scheduled beam-time per (mode x line x machine).

Answers the mentor's question -- "XLEAP HXR on NC has N1 points, XLEAP SXR on NC
has N2" -- from the *schedule* alone: no cluster, no archiver, pure stdlib over
the classified calendar CSVs already in ``data_samples``.

Two important caveats it prints alongside the numbers:

* The unit here is **scheduled shift-hours and event counts, not delivered data
  points.** A booked XLEAP shift that never lased still counts. The true "N data
  points" (1 Hz archiver rows that actually delivered) is Tier B
  (:mod:`census.archiver_census`), which joins these same events against the
  archiver -- HXR only until an SXR pull exists.
* ``primary_class`` is mutually exclusive, so the ``primary`` table's cells sum
  to the total cleanly. Because a shift can wear several mode flags (an XLEAP
  shift also flagged ``two_color``), a second ``flags`` table counts every flag a
  shift carried -- those cells overlap and do **not** sum to the total.

Run::

    python -m census.calendar_census            # print + write default outputs
    python -m census.calendar_census --no-write  # print only
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .common import LINES, MACHINES, MODES, iso_epoch, machine_of

# Repo layout: this file is <repo>/census/, the data lives in the sibling
# workspace dir <repo>/../data_samples/. Resolved once, overridable via CLI.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _REPO_ROOT.parent / "data_samples" / "program_calendar_data" / "cleaned"
_DEFAULT_OUT = _REPO_ROOT.parent / "data_samples" / "census"

_SEC_PER_HOUR = 3600.0


@dataclass(frozen=True)
class Cell:
    """Accumulator for one ``(mode, line, machine)`` census cell."""

    n_events: int = 0
    shift_hours: float = 0.0

    def add(self, hours: float) -> "Cell":
        return Cell(self.n_events + 1, self.shift_hours + hours)


def _classified_csv(data_dir: Path, line: str) -> Path:
    return data_dir / f"{line.lower()}_classified_all.csv"


def tally(data_dir: Path) -> tuple[dict, dict, list[str]]:
    """Tally both the primary-class and the flag-based census over both lines.

    Returns ``(primary, flags, warnings)`` where ``primary``/``flags`` map
    ``(mode, line, machine) -> Cell``. ``primary`` uses the single
    ``primary_class`` label (cells partition the total); ``flags`` fans each
    ``all_flags`` token out into its own cell (cells overlap).
    """
    primary: dict[tuple[str, str, str], Cell] = defaultdict(Cell)
    flags: dict[tuple[str, str, str], Cell] = defaultdict(Cell)
    warnings: list[str] = []

    for line in LINES:
        path = _classified_csv(data_dir, line)
        if not path.exists():
            warnings.append(f"missing {path.name} -- {line} not counted")
            continue
        n_rows = n_bad = 0
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                n_rows += 1
                start_raw, end_raw = row.get("start"), row.get("end")
                if not start_raw or not end_raw:
                    n_bad += 1
                    continue
                try:
                    hours = (iso_epoch(end_raw) - iso_epoch(start_raw)) / _SEC_PER_HOUR
                except ValueError:
                    n_bad += 1
                    continue
                machine = machine_of(row.get("calendar"))
                mode = (row.get("primary_class") or "").strip() or "unlabeled"
                primary[(mode, line, machine)] = primary[(mode, line, machine)].add(hours)
                for token in (row.get("all_flags") or "").split(";"):
                    token = token.strip()
                    if token:
                        key = (token, line, machine)
                        flags[key] = flags[key].add(hours)
        if n_bad:
            warnings.append(f"{path.name}: skipped {n_bad}/{n_rows} rows (bad start/end)")
    return primary, flags, warnings


def _mode_order(cells: dict) -> list[str]:
    """Known modes in canonical order, then any extras seen, alphabetical."""
    seen = {mode for (mode, _line, _machine) in cells}
    extra = sorted(seen - set(MODES))
    return [m for m in MODES if m in seen] + extra


def write_long_csv(primary: dict, flags: dict, out_path: Path) -> None:
    """Write one tidy long-form CSV: one row per non-empty cell, both views."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["view", "mode", "line", "machine", "n_events", "shift_hours"])
        for view, cells in (("primary", primary), ("flags", flags)):
            for (mode, line, machine), cell in sorted(cells.items()):
                writer.writerow(
                    [view, mode, line, machine, cell.n_events, round(cell.shift_hours, 2)]
                )


def _pivot_lines(cells: dict, metric: str) -> list[str]:
    """Render a ``mode`` x ``line/machine`` pivot as aligned text rows."""
    col_keys = [(line, machine) for line in LINES for machine in MACHINES]
    header = ["mode".ljust(16)] + [f"{ln}/{mc}".rjust(11) for ln, mc in col_keys]
    lines_out = ["  ".join(header)]
    for mode in _mode_order(cells):
        row = [mode.ljust(16)]
        for line, machine in col_keys:
            cell = cells.get((mode, line, machine))
            if cell is None:
                row.append("·".rjust(11))
            elif metric == "n_events":
                row.append(str(cell.n_events).rjust(11))
            else:
                row.append(f"{cell.shift_hours:,.0f}".rjust(11))
        lines_out.append("  ".join(row))
    return lines_out


def _print_report(primary: dict, flags: dict, warnings: list[str]) -> None:
    print("=" * 92)
    print("TIER A -- SCHEDULED calendar census  (unit: booked shifts, NOT delivered points)")
    print("=" * 92)
    for title, metric in (("shift-hours", "shift_hours"), ("event count", "n_events")):
        print(f"\n### primary_class -- {title}  (cells partition the total)\n")
        print("\n".join(_pivot_lines(primary, metric)))
    print("\n### all_flags -- event count  (cells OVERLAP; a shift can wear several flags)\n")
    print("\n".join(_pivot_lines(flags, "n_events")))
    if warnings:
        print("\n".join(["\n[warnings]"] + [f"  ! {w}" for w in warnings]))
    print(
        "\nNote: these are SCHEDULED hours/events. Delivered 1 Hz data points per cell "
        "come from Tier A's\n      sibling `census.archiver_census` (HXR only until an SXR "
        "archiver pull exists)."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA,
                        help=f"classified calendar CSV dir (default: {_DEFAULT_DATA})")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT / "calendar_census.csv",
                        help="long-form output CSV path")
    parser.add_argument("--no-write", action="store_true", help="print only, do not write")
    args = parser.parse_args(argv)

    if not args.data_dir.exists():
        print(f"error: data dir not found: {args.data_dir}", file=sys.stderr)
        return 2

    primary, flags, warnings = tally(args.data_dir)
    if not primary:
        print("error: no rows counted (no classified CSVs found)", file=sys.stderr)
        return 1

    _print_report(primary, flags, warnings)
    if not args.no_write:
        write_long_csv(primary, flags, args.out)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
