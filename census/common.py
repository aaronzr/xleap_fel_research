"""Shared, dependency-free primitives for the data census.

Both census tiers agree on three things and they live here so they can never
drift apart:

* how a mode/line/machine *cell* is spelled (:data:`MODES`, :data:`LINES`,
  :data:`MACHINES`);
* how the NC/SC machine is read off a calendar tag (:func:`machine_of`);
* how a timestamp is mapped into the scheduled event that contains it
  (:func:`load_events`, :func:`assign_event`).

Kept stdlib-only on purpose: Tier A (calendar counts) runs on this laptop with
no third-party deps, and Tier B (archiver row counts) imports the *same*
interval-join here so the "which mode was scheduled at time t" rule is identical
on both sides. pandas appears only in :mod:`census.archiver_census`, to read the
S3DF ``.pkl`` files -- it hands the raw epoch index down to :func:`assign_event`.
"""
from __future__ import annotations

import bisect
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

__all__ = [
    "LINES",
    "MODES",
    "MACHINES",
    "Event",
    "machine_of",
    "iso_epoch",
    "load_events",
    "assign_event",
]

# The two undulator lines. Not a data column -- it is implied by which classified
# CSV a row came from (``hxr_classified_all.csv`` vs ``sxr_classified_all.csv``).
LINES: Final[tuple[str, ...]] = ("HXR", "SXR")

# Machine / injector: NC = normal-conducting (Cu, LCLS-I), SC = superconducting
# (LCLS-II). Read off the calendar-tag prefix; "unknown" when absent.
MACHINES: Final[tuple[str, ...]] = ("NC", "SC", "unknown")

# The mutually-exclusive ``primary_class`` values the calendar classifier emits.
# SASE is the residual/complement (no positive flag); ordering is roughly
# most-common first for stable, readable pivot output.
MODES: Final[tuple[str, ...]] = (
    "sase",
    "not_delivering",
    "setup_tune",
    "machine_dev",
    "xleap",
    "self_seed",
    "two_color",
    "two_bunch",
    "harmonic",
    "fresh_slice",
)

_MACHINE_RE: Final[re.Pattern[str]] = re.compile(r"\s*(NC|SC)\b", re.IGNORECASE)


def machine_of(calendar_tag: str | None) -> str:
    """Return ``"NC"``/``"SC"``/``"unknown"`` from a calendar tag string.

    The tag looks like ``"NC-TMO"`` / ``"SC-CXI"``; the machine is its leading
    token. Anything without a leading ``NC``/``SC`` (blank tags, free text) is
    ``"unknown"`` so it is counted, never silently dropped.
    """
    if not calendar_tag:
        return "unknown"
    match = _MACHINE_RE.match(calendar_tag)
    return match.group(1).upper() if match else "unknown"


def iso_epoch(stamp: str) -> float:
    """Parse an ISO-8601 timestamp (with offset or ``Z``) to POSIX seconds (UTC).

    Calendar times carry an explicit offset (``...-08:00``); we normalise to a
    single epoch axis so a shift's duration and any interval test are timezone-
    agnostic. ``Z`` is accepted for robustness against future raw pulls.
    """
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


@dataclass(frozen=True, order=True)
class Event:
    """One scheduled program shift, reduced to what the census needs.

    ``start``/``end`` are POSIX seconds (UTC). ``order=True`` with ``start`` first
    means a list of events sorts chronologically, which :func:`assign_event`
    relies on for its bisect. ``mode`` is the ``primary_class``; ``machine`` is
    NC/SC/unknown; ``line`` is HXR/SXR.
    """

    start: float
    end: float
    mode: str
    machine: str
    line: str


def load_events(csv_path: str | Path, line: str) -> list[Event]:
    """Load one classified calendar CSV into ``start``-sorted :class:`Event` rows.

    ``line`` labels every event (the CSV does not carry it). Rows with an empty
    or unparseable ``start``/``end`` are skipped with no row dropped silently
    elsewhere -- the caller can compare ``len`` against the file's line count.
    """
    events: list[Event] = []
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            start_raw, end_raw = row.get("start"), row.get("end")
            if not start_raw or not end_raw:
                continue
            try:
                start, end = iso_epoch(start_raw), iso_epoch(end_raw)
            except ValueError:
                continue
            events.append(
                Event(
                    start=start,
                    end=end,
                    mode=(row.get("primary_class") or "").strip(),
                    machine=machine_of(row.get("calendar")),
                    line=line,
                )
            )
    events.sort()
    return events


def assign_event(events: list[Event], epoch: float) -> Event | None:
    """Return the scheduled event covering ``epoch`` (POSIX s), or ``None``.

    ``events`` must be sorted by ``start`` (as :func:`load_events` returns). We
    bisect to the last event that started at or before ``epoch`` and walk back
    while starts still precede it, returning the first whose ``end`` covers the
    instant. This tolerates the occasional back-to-back / slightly overlapping
    shift without assuming a strict partition of the timeline; a gap (no shift
    scheduled) yields ``None`` -- the caller counts those as an ``unscheduled``
    cell rather than forcing them into a mode.
    """
    if not events:
        return None
    # Probe sorts strictly after every real event whose start == epoch (no real
    # shift has an infinite end), so bisect lands just past them and idx is the
    # last event with start <= epoch -- boundary-inclusive on the start side.
    probe = Event(epoch, float("inf"), "￿", "￿", "￿")
    idx = bisect.bisect_right(events, probe) - 1
    while idx >= 0 and events[idx].start <= epoch:
        if epoch < events[idx].end:
            return events[idx]
        idx -= 1
    return None
