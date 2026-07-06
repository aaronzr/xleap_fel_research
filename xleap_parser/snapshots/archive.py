"""Fetch + bin PV history via the MEME archive service.

The pure-logic "library" layer (the analog of a C translation unit): it takes a
:class:`~snapshots.config.RunConfig` and returns rows. No argv parsing, no
``print``, no file writes -- those belong to :mod:`snapshots.__main__`. That
separation is what makes this unit-testable.
"""
from __future__ import annotations

import calendar
import math
import statistics
import time
from datetime import datetime, timezone

from .config import RunConfig


def iso_utc(secs: int) -> str:
    """UTC wall-clock ISO (naive, no tz suffix) for an epoch-seconds value."""
    return (
        datetime.fromtimestamp(secs, tz=timezone.utc).replace(tzinfo=None).isoformat()
    )


def window_epoch(from_time: str) -> int:
    """Epoch seconds of the window start.

    The archiver prepends one at-or-before boundary sample (~1s before the start)
    that we drop so it doesn't make a pre-window row.
    """
    return calendar.timegm(time.strptime(from_time[:19], "%Y-%m-%dT%H:%M:%S"))


def bucket_start(s: int, from_epoch: int, bin_seconds: int) -> int:
    """Bin start for sample-epoch ``s``, anchored to the window start.

    Anchoring to ``from_epoch`` (not the global 1970 grid) lines the bin labels up
    with ``from_time`` for any window -- including short or grid-unaligned ones, so
    the first bin's nominal time IS ``from_time`` and its motion window is measured
    from there. Identical to global-grid flooring when ``from_time`` sits on a
    ``bin_seconds`` boundary (e.g. the 900 s snapshot grid).
    """
    return from_epoch + ((s - from_epoch) // bin_seconds) * bin_seconds


def utc_datetime(iso_zulu: str) -> datetime:
    """Parse an ISO8601 Zulu string to a UTC-*aware* datetime.

    We hand meme.archive.get a datetime whose ``tzname()`` is ``"UTC"`` rather
    than the raw string. meme parses strings with ``dateparser``, which yields an
    aware datetime whose tzname is not literally ``"UTC"``; that trips meme's
    guard into ``convert_datetime_to_UTC`` on an already-aware value and raises
    "Not naive datetime (tzinfo is already set)". A ``timezone.utc`` datetime
    satisfies meme's UTC/GMT check, so it skips that broken path.
    """
    return datetime.strptime(iso_zulu[:19], "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc
    )


def _series(data) -> tuple:
    """Normalize a single-PV ``meme.archive.get`` payload to (secs, values).

    Unwraps the nested ``value`` layers that differ between the HTTP and PVA
    transports until the leaf columns are found.
    """
    node = data
    for _ in range(4):
        if isinstance(node, dict) and "secondsPastEpoch" in node and "values" in node:
            return node["secondsPastEpoch"], node["values"]
        if isinstance(node, dict) and "value" in node:
            node = node["value"]
        else:
            break
    shape = list(data) if isinstance(data, dict) else type(data).__name__
    raise ValueError(f"unrecognized archive payload shape: {shape}")


def _get_series(query: str, cfg: RunConfig) -> tuple:
    """Run one ``meme.archive.get`` and normalize it to ``(secs, values)``.

    ``meme.archive`` is imported lazily so the pure motion helpers below (and
    their tests) do not need the archive client installed.
    """
    import meme.archive

    data = meme.archive.get(
        query,
        from_time=utc_datetime(cfg.from_time),
        to_time=utc_datetime(cfg.to_time),
        timeout=cfg.timeout,
        protocol=cfg.protocol,
    )
    return _series(data)


def _window_moved(values: list[float], threshold: float) -> bool:
    """Did a PV move within one snapshot window?

    ``moved`` when the fractional spread ``(max - min) / |median|`` over the
    window's raw samples exceeds ``threshold``. A window with <= 1 sample cannot
    show a spread, so it is stationary (Aaron: a unique value in the interval =>
    stationary / thermal noise, regardless of archive cadence).
    """
    if len(values) <= 1:
        return False
    median = statistics.median(values)
    spread = max(values) - min(values)
    if median == 0.0:  # guard the (physically impossible for K) zero-median case
        return spread > threshold
    return spread / abs(median) > threshold


def _fractional_spread(values: list[float]) -> float:
    """Fractional spread ``(max - min) / |median|`` over a bin's raw samples.

    <= 1 sample cannot show a spread, so it is ``0.0`` (stationary). The
    (physically impossible for K/energy) zero-median case returns the raw span.
    """
    if len(values) <= 1:
        return 0.0
    median = statistics.median(values)
    span = max(values) - min(values)
    return span if median == 0.0 else span / abs(median)


def _spread_by_bucket(secs_arr, vals_arr, cfg: RunConfig) -> dict[int, float]:
    """Map each bin's start epoch -> fractional spread over the FULL bin.

    Unlike motion (which inspects only the first ``snapshot_delta_s`` of the bin),
    the spread uses *every* sample in the bin -- so a bin sitting on a slow,
    minutes-long excursion (e.g. a dump-magnet standardization swinging the beam
    energy readback) shows a large spread even though its first few seconds looked
    steady. Consumed by the analysis' energy-stability gate.
    """
    from_epoch = window_epoch(cfg.from_time)
    bins: dict[int, list[float]] = {}
    for secs, val in zip(secs_arr, vals_arr):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        s = int(secs)
        if s < from_epoch:  # drop the prepended pre-window boundary sample
            continue
        bucket = bucket_start(s, from_epoch, cfg.bin_seconds)
        bins.setdefault(bucket, []).append(float(val))
    return {bucket: _fractional_spread(vals) for bucket, vals in bins.items()}


def _motion_by_bucket(secs_arr, vals_arr, cfg: RunConfig) -> dict[int, bool]:
    """Map each bin's start epoch -> whether the PV moved within ``[t, t+delta]``.

    ``secs_arr``/``vals_arr`` are the RAW (unbinned) samples for one PV over the
    whole window. Each sample is assigned to its bin, but only samples landing
    within ``snapshot_delta_s`` of the bin start count toward that bin's motion
    window -- equivalent to a per-nominal-time raw fetch over ``[t, t+delta]``,
    done as one stream so we pay one round-trip per PV instead of one per bin.
    """
    from_epoch = window_epoch(cfg.from_time)
    windows: dict[int, list[float]] = {}
    for secs, val in zip(secs_arr, vals_arr):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        s = int(secs)
        if s < from_epoch:  # drop the prepended pre-window boundary sample
            continue
        bucket = bucket_start(s, from_epoch, cfg.bin_seconds)
        if s - bucket <= cfg.snapshot_delta_s:
            windows.setdefault(bucket, []).append(float(val))
    return {
        bucket: _window_moved(vals, cfg.motion_threshold)
        for bucket, vals in windows.items()
    }


def _fetch_bin_stats(
    pv: str, cfg: RunConfig
) -> tuple[dict[int, bool], dict[int, float]]:
    """Per-bin ``(moved, spread)`` maps for ``pv`` from one raw (no-operator) fetch.

    ``moved`` is the snapshot-instant motion (first ``snapshot_delta_s`` of the
    bin); ``spread`` is the fractional range over the whole bin. Both come from
    the same raw pull, so there is one extra archive round-trip per PV, not two.
    Returns empty maps (all stationary / unknown) when the probe is disabled or
    the raw fetch fails -- a probe must never drop or mask a value row.
    """
    if cfg.snapshot_delta_s <= 0:
        return {}, {}
    try:
        secs_arr, vals_arr = _get_series(pv, cfg)  # bare PV name => raw samples
    except Exception:  # noqa: BLE001 - a probe failure just means "unknown"
        return {}, {}
    if secs_arr is None or len(secs_arr) == 0:
        return {}, {}
    return (
        _motion_by_bucket(secs_arr, vals_arr, cfg),
        _spread_by_bucket(secs_arr, vals_arr, cfg),
    )


def fetch_pv(pv: str, cfg: RunConfig) -> tuple[str, list, str]:
    """Fetch one PV -> ``(nominal_time, pv, timestamp, value, moved, spread)`` rows.

    The value pass bins the operator-reduced samples (first sample per bin); the
    probe pass adds, per bin, ``moved`` (PV moved within ``snapshot_delta_s`` of the
    nominal time -- Aaron's rule) and ``spread`` (fractional range over the whole
    bin, for the energy-stability gate). Returns ``(pv, rows, status)``; status is
    'ok', 'empty', or 'error: ...'.
    """
    from_epoch = window_epoch(cfg.from_time)
    query = f"{cfg.bin_operator}({pv})" if cfg.bin_operator else pv
    try:
        secs_arr, vals_arr = _get_series(query, cfg)
    except IndexError:
        # meme's HTTP handler indexes ``pvdata[0]`` on an empty archiver response,
        # so an IndexError here means "no data for this PV in this window" (e.g. the
        # PV did not exist yet) -- a legitimate empty, not a fetch error.
        return pv, [], "empty"
    except Exception as exc:  # noqa: BLE001 - report any I/O/parse failure per PV
        return pv, [], f"error: {exc}"

    if secs_arr is None or len(secs_arr) == 0:
        return pv, [], "empty"

    # Keep the FIRST sample per bucket = value as of the bin's nominal time.
    by_bucket: dict[int, tuple[int, float]] = {}
    for secs, val in zip(secs_arr, vals_arr):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        s = int(secs)
        if s < from_epoch:  # drop the prepended pre-window boundary sample
            continue
        bucket = bucket_start(s, from_epoch, cfg.bin_seconds)
        if bucket not in by_bucket:  # data is time-ordered; first wins
            by_bucket[bucket] = (s, val)

    moved_by_bucket, spread_by_bucket = _fetch_bin_stats(pv, cfg)
    rows = [
        (
            iso_utc(bucket), pv, iso_utc(s), float(val),
            moved_by_bucket.get(bucket, False),
            spread_by_bucket.get(bucket, float("nan")),
        )
        for bucket, (s, val) in sorted(by_bucket.items())
    ]
    return pv, rows, "ok" if rows else "empty"
