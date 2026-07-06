"""CSV-backed time-series cache with coverage tracking.

The notebook and ``python -m snapshots`` already treat a CSV as the datastore of
record (``snapshots.csv``): a long table of ``(nominal_time, ..., value)`` rows.
This cache keeps that spirit -- one CSV per *kind* of series (raw PV values,
derived taper, derived n_und, ...) -- and adds the one thing a server needs that
a one-shot script does not: knowing *which time windows it has already pulled* so
a repeat request can be served without calling MEME again.

Two files back a cache instance:

    <name>.csv           long rows: key, nominal_time, <value columns...>
    <name>.coverage.json key -> merged list of covered [from, to) ISO ranges

``key`` bundles everything that changes the numbers (the PV or quantity, the
snapshot ``window``, and the ``interval``), so two requests that differ only in
their time span share cached rows, while a request at a different resolution gets
its own bucket. Coverage is tracked separately from the rows because an empty
bin legitimately produces no row -- "covered but no data" must be
distinguishable from "never fetched".
"""
from __future__ import annotations

import csv
import json
import threading
from datetime import datetime
from pathlib import Path

__all__ = ["TimeSeriesCache", "cache_key", "parse_dt", "iso_naive"]

# The nominal_time / range strings we store are naive-UTC ISO seconds, matching
# what ``snapshots.iso_utc`` emits (e.g. "2026-06-01T00:00:00"). Keeping one
# canonical spelling lets us compare ranges by parsing to datetime.
_KEY_COLUMN = "key"
_TIME_COLUMN = "nominal_time"


def iso_naive(dt: datetime) -> str:
    """Canonical naive-UTC ISO-seconds spelling used for stored timestamps."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(tz=None).replace(tzinfo=None)
    return dt.replace(microsecond=0).isoformat()


def parse_dt(value: str) -> datetime:
    """Parse an ISO8601 string (optionally ``Z``-suffixed) to naive UTC.

    Accepts the ``Z`` Zulu suffix (which ``datetime.fromisoformat`` rejects on
    older Pythons) and normalises any aware value to naive UTC so all cached and
    requested timestamps live on one comparable scale.
    """
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is not None:
        dt = dt.astimezone(tz=None).replace(tzinfo=None)
    return dt.replace(microsecond=0) if dt.microsecond else dt


def cache_key(ident: str, *, window: float, interval: int) -> str:
    """Bucket rows by the parameters that change their values.

    ``ident`` names the PV or quantity; ``window``/``interval`` are the snapshot
    motion window and nominal-grid spacing. Requests that differ only in their
    time span collapse to the same key and reuse each other's rows.
    """
    return f"{ident}|w={float(window):g}|i={int(interval)}"


def _merge_ranges(ranges: list[list[str]]) -> list[list[str]]:
    """Coalesce overlapping/adjacent ``[from, to)`` ISO ranges, sorted."""
    parsed = sorted(
        ([parse_dt(a), parse_dt(b)] for a, b in ranges), key=lambda r: r[0]
    )
    merged: list[list[datetime]] = []
    for lo, hi in parsed:
        if merged and lo <= merged[-1][1]:  # overlaps or touches the last range
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [[iso_naive(lo), iso_naive(hi)] for lo, hi in merged]


class TimeSeriesCache:
    """A long CSV of keyed time-series rows plus a per-key coverage manifest.

    Thread-safe (a single lock guards the in-memory rows + coverage and their
    CSV/JSON files) so the threaded HTTP server can share one instance across
    request handlers.
    """

    def __init__(self, path: str | Path, value_columns: list[str]) -> None:
        self.path = Path(path)
        self.value_columns = list(value_columns)
        self._columns = [_KEY_COLUMN, _TIME_COLUMN, *self.value_columns]
        self._coverage_path = self.path.with_suffix(self.path.suffix + ".coverage.json")
        self._lock = threading.Lock()
        # rows[key] -> list of dict(nominal_time=..., <value columns...>)
        self._rows: dict[str, list[dict[str, str]]] = {}
        self._coverage: dict[str, list[list[str]]] = {}
        self._load()

    # --- persistence ---------------------------------------------------------
    def _load(self) -> None:
        if self.path.exists():
            with self.path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    key = row.get(_KEY_COLUMN, "")
                    record = {c: row.get(c, "") for c in [_TIME_COLUMN, *self.value_columns]}
                    self._rows.setdefault(key, []).append(record)
        if self._coverage_path.exists():
            with self._coverage_path.open() as fh:
                self._coverage = json.load(fh)

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(self._columns)
            for key in sorted(self._rows):
                for record in self._rows[key]:
                    writer.writerow([key, *(record.get(c, "") for c in [_TIME_COLUMN, *self.value_columns])])
        with self._coverage_path.open("w") as fh:
            json.dump(self._coverage, fh, indent=2, sort_keys=True)

    # --- coverage ------------------------------------------------------------
    def covered(self, key: str, start: datetime, end: datetime) -> bool:
        """Is the whole ``[start, end)`` span already fetched for ``key``?"""
        with self._lock:
            for lo, hi in self._coverage.get(key, []):
                if parse_dt(lo) <= start and end <= parse_dt(hi):
                    return True
        return False

    # --- reads ---------------------------------------------------------------
    def get(self, key: str, start: datetime, end: datetime) -> list[dict[str, str]]:
        """Cached rows for ``key`` whose nominal_time is in ``[start, end)``."""
        with self._lock:
            rows = [
                dict(record)
                for record in self._rows.get(key, [])
                if start <= parse_dt(record[_TIME_COLUMN]) < end
            ]
        rows.sort(key=lambda r: r[_TIME_COLUMN])
        return rows

    # --- writes --------------------------------------------------------------
    def put(
        self,
        key: str,
        start: datetime,
        end: datetime,
        rows: list[dict[str, object]],
    ) -> None:
        """Store ``rows`` for ``key`` and record ``[start, end)`` as covered.

        Rows already present at the same nominal_time are replaced, so a refetch
        of an overlapping span updates values rather than duplicating them. The
        covered range is recorded even when ``rows`` is empty -- an empty span is
        a legitimate "we looked, there was nothing" result.
        """
        with self._lock:
            existing = {r[_TIME_COLUMN]: r for r in self._rows.get(key, [])}
            for row in rows:
                record = {_TIME_COLUMN: str(row[_TIME_COLUMN])}
                record.update({c: _to_cell(row.get(c, "")) for c in self.value_columns})
                existing[record[_TIME_COLUMN]] = record
            self._rows[key] = sorted(existing.values(), key=lambda r: r[_TIME_COLUMN])
            spans = self._coverage.get(key, [])
            spans.append([iso_naive(start), iso_naive(end)])
            self._coverage[key] = _merge_ranges(spans)
            self._flush()


def _to_cell(value: object) -> str:
    """Serialise a cell for CSV, keeping bools/NaN round-trippable as text."""
    if isinstance(value, bool):
        return "True" if value else "False"
    return "" if value is None else str(value)
