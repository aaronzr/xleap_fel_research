"""Drive `python -m snapshots` over every SXR XLEAP calendar window.

Runs **on the SLAC network** (e.g. ``dev-epics``), where ``meme.archive`` and the
EPICS environment are available -- the fetch itself talks to the MEME archive
service. This script only orchestrates: it reads the calendar's SXR-XLEAP event
list, converts each block to UTC, and shells out to the already-tested
``snapshots`` fetcher once per window, writing one CSV per event.

One-CSV-per-event is deliberate: each file is a self-contained XLEAP session, so
the downstream time-based train/val/test split is just a partition of files by
date -- no re-slicing of a merged blob.

Everything here is stdlib-only (no numpy/pandas): it must import cleanly in the
minimal on-cluster environment whose only heavy dep is ``meme`` (pulled in lazily
by the fetcher, not by this driver).

Typical use (from this directory, on dev-epics)::

    # dry-run: print the commands that WOULD run, pull nothing
    python pull_sxr.py --events sxr_xleap.csv --out-dir pulls/sxr --dry-run

    # real pull, first 20 windows (batch across sessions with --start/--limit)
    python pull_sxr.py --events sxr_xleap.csv --out-dir pulls/sxr --limit 20

    # resume: already-pulled windows are skipped unless --overwrite
    python pull_sxr.py --events sxr_xleap.csv --out-dir pulls/sxr

The event CSV is the calendar export (columns ``start,end,...`` with Pacific
ISO8601 offsets, e.g. ``2018-02-07T09:00:00-08:00``). Copy it to the remote
alongside the code before running.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# This script lives in ``xleap_parser/scripts/``; the parser root (one level up)
# is where the ``snapshots`` package + ``beamlines`` live, so the fetcher subprocess
# is run with that as its CWD regardless of where the user invokes this script.
PARSER_ROOT = Path(__file__).resolve().parent.parent


def to_utc_zulu(iso_with_offset: str) -> str:
    """Calendar timestamp (Pacific, offset-aware) -> ISO8601 UTC 'Zulu' string.

    The event CSV stores wall-clock Pacific times WITH their UTC offset
    (``-08:00`` winter / ``-07:00`` summer). The ``snapshots`` fetcher parses only
    the first 19 chars of ``--from``/``--to`` and *forces* UTC, so we must hand it
    the true UTC wall-clock -- otherwise every window is off by 7-8 h. Converting
    here (offset-aware -> UTC) is the single source of truth for that shift.
    """
    dt = datetime.fromisoformat(iso_with_offset.strip())
    if dt.tzinfo is None:
        raise ValueError(
            f"event time {iso_with_offset!r} has no UTC offset; refusing to guess "
            "the timezone (would risk a silent 7-8 h shift)"
        )
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def pad_window(start_z: str, end_z: str, pad_minutes: int) -> tuple[str, str]:
    """Widen a [start, end] UTC-Zulu window by ``pad_minutes`` on each side.

    Scheduled block boundaries are approximate; a small pad catches lasing that
    begins during setup just before (or trails just after) the program slot.
    """
    if pad_minutes <= 0:
        return start_z, end_z
    fmt = "%Y-%m-%dT%H:%M:%S.000Z"
    s = datetime.strptime(start_z, fmt).replace(tzinfo=timezone.utc)
    e = datetime.strptime(end_z, fmt).replace(tzinfo=timezone.utc)
    from datetime import timedelta

    s -= timedelta(minutes=pad_minutes)
    e += timedelta(minutes=pad_minutes)
    return s.strftime(fmt), e.strftime(fmt)


def out_name(index: int, start_z: str) -> str:
    """Deterministic per-event filename: ``sxr_<NNN>_<YYYYMMDDTHHMMSS>.csv``.

    Zero-padded index keeps lexical order == chronological order (events are
    already sorted ascending in the calendar export), and the embedded UTC start
    makes each file self-describing for the later date-based split.
    """
    stamp = start_z[:19].replace("-", "").replace(":", "")
    return f"sxr_{index:03d}_{stamp}.csv"


def read_events(path: Path) -> list[tuple[str, str]]:
    """Read (start, end) offset-aware ISO strings from the calendar event CSV.

    Tolerates space-padded / column-aligned exports by stripping every header and
    cell, so a header like ``'start          '`` still matches ``'start'`` and a
    padded value still parses.
    """
    with open(path, newline="") as fh:
        rows = [[cell.strip() for cell in r] for r in csv.reader(fh)]
    if not rows:
        raise ValueError(f"{path}: empty file")
    header = rows[0]
    try:
        i_start, i_end = header.index("start"), header.index("end")
    except ValueError:
        raise ValueError(
            f"{path}: expected 'start' and 'end' columns, got {header}"
        ) from None
    events: list[tuple[str, str]] = []
    for r in rows[1:]:
        if len(r) > max(i_start, i_end) and r[i_start] and r[i_end]:
            events.append((r[i_start], r[i_end]))
    return events


def build_cmd(
    pvs: str | None, start_z: str, end_z: str, out_path: Path, protocol: str,
    bin_seconds: int, jobs: int, timeout: float,
) -> list[str]:
    """The exact ``python -m snapshots`` invocation for one window.

    ``pvs is None`` lets the fetcher use its own line-anchored default
    (``beamlines.DEFAULT_LINE.pvs_csv`` -> ``config/pvs_<line>.csv``); pass a path
    only to override it. ``out_path`` must be absolute -- the subprocess runs with
    CWD = parser root, not the user's CWD.
    """
    return [
        sys.executable, "-m", "snapshots",
        *(("-p", pvs) if pvs else ()),
        "--from", start_z,
        "--to", end_z,
        "--protocol", protocol,
        "--bin-seconds", str(bin_seconds),
        "--timeout", str(timeout),
        "-j", str(jobs),
        "-o", str(out_path),
    ]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python pull_sxr.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--events", default="data/calendar/cleaned/sxr_xleap.csv",
                   help="calendar event CSV (start,end,... cols); path is relative to CWD. "
                        "default: data/calendar/cleaned/sxr_xleap.csv (run from the repo root)")
    p.add_argument("--pvs", default=None,
                   help="override the PV list; default None = the fetcher's line-anchored "
                        "config/pvs_<line>.csv")
    p.add_argument("--out-dir", default="data/pulls/sxr",
                   help="directory for per-event CSVs + manifest; relative to CWD. "
                        "default: data/pulls/sxr")
    p.add_argument("--protocol", default="HTTP", choices=["PVA", "HTTP"],
                   help="MEME transport. default: HTTP (REST to the archive appliance; "
                        "avoids the PVA 'hist' name-resolution collision on dev nodes)")
    p.add_argument("--bin-seconds", type=int, default=900,
                   help="nominal grid / bin width (s). default: 900")
    p.add_argument("-j", "--jobs", type=int, default=16,
                   help="concurrent PV requests per window (lower = gentler on MEME). default: 16")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="per-request timeout (s). default: 60")
    p.add_argument("--pad-minutes", type=int, default=0,
                   help="widen each window by N min per side to catch setup lasing. default: 0")
    p.add_argument("--start", type=int, default=0,
                   help="0-based event index to start at (batch across sessions). default: 0")
    p.add_argument("--limit", type=int, default=None,
                   help="pull at most N windows this run. default: all remaining")
    p.add_argument("--sleep-between", type=float, default=0.0,
                   help="pause N seconds between windows to ease archiver load. default: 0")
    p.add_argument("--overwrite", action="store_true",
                   help="re-pull windows whose output already exists (default: skip = resume)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands that would run; pull nothing")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Resolve relative paths against the current working directory, like any CLI
    # tool -- NOT against the script's own folder. ``out_dir`` is made ABSOLUTE
    # because the fetcher subprocess runs with a different CWD (the parser root),
    # so a relative ``-o`` would otherwise land in the wrong place.
    events_path = Path(args.events)
    events = read_events(events_path)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lo = max(args.start, 0)
    hi = len(events) if args.limit is None else min(lo + args.limit, len(events))
    selected = list(enumerate(events))[lo:hi]

    print(
        f"SXR XLEAP pull: {len(events)} events total, running [{lo}:{hi}] "
        f"({len(selected)} windows)  proto={args.protocol}  bin={args.bin_seconds}s  "
        f"pad={args.pad_minutes}m  -> {out_dir}"
    )

    manifest_path = out_dir / "manifest.csv"
    new_manifest = not manifest_path.exists()
    n_ok = n_skip = n_empty = n_err = 0

    with open(manifest_path, "a", newline="") as mf:
        writer = csv.writer(mf)
        if new_manifest:
            writer.writerow(
                ["index", "start_utc", "end_utc", "out_file", "status", "rows"]
            )

        for index, (start_raw, end_raw) in selected:
            try:
                start_z, end_z = to_utc_zulu(start_raw), to_utc_zulu(end_raw)
            except ValueError as exc:
                print(f"  [{index}] SKIP bad time: {exc}", file=sys.stderr)
                writer.writerow([index, start_raw, end_raw, "", "error: bad time", 0])
                n_err += 1
                continue
            start_z, end_z = pad_window(start_z, end_z, args.pad_minutes)

            out_path = out_dir / out_name(index, start_z)
            # Resume on DATA, not mere existence: a header-only file from a failed
            # run (0 data rows) must be re-pulled, not treated as done.
            if out_path.exists() and _count_rows(out_path) > 0 and not args.overwrite:
                print(f"  [{index}] skip (has data): {out_path.name}")
                n_skip += 1
                continue

            cmd = build_cmd(
                args.pvs, start_z, end_z, out_path, args.protocol,
                args.bin_seconds, args.jobs, args.timeout,
            )
            if args.dry_run:
                print("  " + " ".join(cmd))
                continue

            print(f"  [{index}] {start_z} .. {end_z} -> {out_path.name}")
            proc = subprocess.run(cmd, cwd=PARSER_ROOT)  # so `-m snapshots` + beamlines resolve
            rows = _count_rows(out_path)
            if proc.returncode != 0:
                status = f"error: exit {proc.returncode}"
                n_err += 1
            elif rows == 0:
                status = "empty"
                n_empty += 1
            else:
                status = "ok"
                n_ok += 1
            writer.writerow([index, start_z, end_z, out_path.name, status, rows])
            mf.flush()
            if args.sleep_between > 0:
                time.sleep(args.sleep_between)

    if args.dry_run:
        print("\nDry run: no data pulled.")
        return 0
    print(
        f"\nDone. ok={n_ok}  empty={n_empty}  errors={n_err}  skipped={n_skip}  "
        f"manifest -> {manifest_path}"
    )
    return 1 if n_err else 0


def _count_rows(path: Path) -> int:
    """Data-row count (excludes header); 0 if the file is missing/empty."""
    if not path.exists():
        return 0
    with open(path, newline="") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


if __name__ == "__main__":
    raise SystemExit(main())
