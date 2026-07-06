"""Compute layer: turn a time window into PV series and derived quantities.

The pure-logic bridge between the HTTP layer and the existing analysis code. It
owns *no* caching and *no* argv/JSON handling -- it takes a window plus the
snapshot ``window``/``interval`` knobs and returns plain row dicts, calling
:func:`snapshots.fetch_pv` for raw archive pulls and :func:`taper.xleap_timeline`
for the derived taper / lasing-group verdicts.

The archive fetcher is injected (defaults to the real ``snapshots.fetch_pv``) so
the server can be exercised end-to-end against a synthetic archive with no live
MEME connection -- the same library/main testability split the rest of the repo
uses.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import pandas as pd

from snapshots import RunConfig
from snapshots.__main__ import read_pvs
from snapshots.archive import fetch_pv as _real_fetch_pv
from taper import DEFAULT_LINE, Beamline, DetectionParams, SnapshotStore, xleap_timeline
from taper.service import DEFAULT_ENERGY_SPREAD_MAX
from taper.store import _finalize

from .cache import iso_naive, parse_dt

__all__ = ["Backend", "PV_COLUMNS", "DERIVED_COLUMNS"]

# Row schemas the compute layer emits (and the cache stores).
PV_COLUMNS = ["timestamp", "value", "moved", "spread"]
DERIVED_COLUMNS = ["timestamp", "xleap_on", "n_und", "taper", "moving", "energy_unsteady"]

FetchFn = Callable[[str, RunConfig], "tuple[str, list, str]"]


def _iso_zulu(dt: datetime) -> str:
    """RunConfig wants a ``...Z`` Zulu string; give it one from a naive-UTC dt."""
    return iso_naive(dt) + "Z"


@dataclass
class Backend:
    """Runs archive fetches and the taper analysis for a request window.

    ``pvs`` defaults to the active line's PV list (``pvs_<line>.csv``) -- the same
    file the fetch CLI reads -- so the derived endpoints see exactly the undulator
    K and beam-energy PVs the notebook uses. ``fetch`` is the archive call; inject
    a stand-in in tests to avoid a live MEME connection.
    """

    line: Beamline = DEFAULT_LINE
    fetch: FetchFn = _real_fetch_pv
    pvs: list[str] = field(default_factory=list)
    jobs: int = 8
    params: DetectionParams = field(default_factory=DetectionParams)
    energy_spread_max: float = DEFAULT_ENERGY_SPREAD_MAX

    def __post_init__(self) -> None:
        if not self.pvs:
            self.pvs = read_pvs(self.line.pvs_csv)

    # --- config --------------------------------------------------------------
    def run_config(
        self, start: datetime, end: datetime, interval: int, window: float
    ) -> RunConfig:
        """A :class:`RunConfig` whose bins anchor at ``start`` (see archive.bucket_start).

        ``interval`` becomes the nominal-grid ``bin_seconds`` and ``window`` the
        ``snapshot_delta_s`` motion window, so "motion within the snapshot window"
        is measured over ``[t, t+window]`` of every ``interval``-spaced bin.
        """
        return RunConfig(
            from_time=_iso_zulu(start),
            to_time=_iso_zulu(end),
            bin_seconds=int(interval),
            snapshot_delta_s=float(window),
        )

    # --- raw PV series -------------------------------------------------------
    def pv_series(
        self, pv: str, start: datetime, end: datetime, interval: int, window: float
    ) -> list[dict[str, object]]:
        """Timestamped values for one PV over ``[start, end)``.

        Each row carries the sample ``timestamp`` and ``value`` plus the snapshot
        probe's ``moved`` (did the PV move within ``window`` of the nominal time)
        and ``spread`` (fractional range over the whole bin) flags.
        """
        cfg = self.run_config(start, end, interval, window)
        _pv, rows, status = self.fetch(pv, cfg)
        if status.startswith("error"):
            raise ArchiveError(f"archive fetch failed for {pv}: {status}")
        return [
            {
                "nominal_time": nominal_time,
                "timestamp": timestamp,
                "value": value,
                "moved": moved,
                "spread": spread,
            }
            for (nominal_time, _p, timestamp, value, moved, spread) in rows
        ]

    # --- derived quantities --------------------------------------------------
    def _fetch_all(self, start: datetime, end: datetime, interval: int, window: float):
        """Fetch every configured PV once and assemble a :class:`SnapshotStore`.

        Undulator K and beam-energy PVs are pulled concurrently (one archive
        round-trip each, like ``python -m snapshots``); the long rows are stitched
        into the wide store the taper analysis expects.
        """
        cfg = self.run_config(start, end, interval, window)
        all_rows: list[tuple] = []
        with ThreadPoolExecutor(max_workers=max(1, self.jobs)) as pool:
            for _pv, rows, status in pool.map(lambda p: self.fetch(p, cfg), self.pvs):
                if status == "ok":
                    all_rows.extend(rows)
        frame = pd.DataFrame(
            all_rows,
            columns=["nominal_time", "pv", "timestamp", "value", "moved", "spread"],
        )
        return SnapshotStore(_finalize(frame))

    def derived_rows(
        self, start: datetime, end: datetime, interval: int, window: float
    ) -> list[dict[str, object]]:
        """Per-nominal-time taper and lasing verdicts over ``[start, end)``.

        Runs the full notebook analysis: builds the K matrix + gamma series, then
        for each nominal time reports ``taper``, ``n_und`` (number of lasing
        undulators) and ``xleap_on``. Time points where an undulator moved within
        the snapshot window (``moving``) or the beam energy was unsteady across the
        bin (``energy_unsteady``) are force-cleared -- taper NaN, ``n_und`` 0 --
        exactly as :func:`taper.xleap_timeline` does, so those points are skipped
        for the taper / lasing-group calculation while staying visible in the
        timeline.
        """
        store = self._fetch_all(start, end, interval, window)
        points = xleap_timeline(
            store,
            line=self.line,
            params=self.params,
            start=start,
            end=end,
            energy_spread_max=self.energy_spread_max,
        )
        return [
            {
                "nominal_time": iso_naive(point.datetime),
                "timestamp": iso_naive(point.datetime),
                "xleap_on": point.xleap_on,
                "n_und": point.n_und,
                "taper": point.taper,
                "moving": point.moving,
                "energy_unsteady": point.energy_unsteady,
            }
            for point in points
        ]


class ArchiveError(RuntimeError):
    """A live archive fetch failed (bubbles up to a 502 at the HTTP layer)."""
