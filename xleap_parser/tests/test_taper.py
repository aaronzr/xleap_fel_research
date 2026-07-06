"""Smoke + unit tests for the `taper` package.

Runs standalone (``python tests/test_taper.py``) or under pytest. Synthesizes a
small long-form snapshot CSV with a planted hockey-stick taper and checks the
whole chain: store pivot -> lasing detection -> per-time XLEAP timeline.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root importable so `taper` / `snapshots` resolve when run from tests/
# (both `pytest` and `python tests/test_taper.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taper import (  # noqa: E402 -- imported after the sys.path shim above
    DEFAULT_LINE as LINE,
    BEAM_MOMENTUM_PV,
    KACT_PV_PATTERN,
    DetectionParams,
    SnapshotStore,
    dk_to_dgamma,
    gamma_from_momentum_gev,
    lasing_mask,
    taper_mev_per_fs,
    timeline_frame,
    xleap_timeline,
)

UND_NUMBERS = [str(1450 + 100 * i) for i in range(10)]  # 1450..2350
MOMENTUM_GEV = 4.0
STEP = 0.05  # per-undulator K rise inside the planted group (>> 4*rho)


def _synthetic_csv(path, moving_at: set[int] | None = None) -> None:
    """Write a long CSV: flat K on unds 0-2, a rising ramp on unds 3-9.

    With ``moving_at is None`` the CSV omits the ``moved`` column entirely (the
    legacy schema, exercising the store's default-False fallback). Otherwise a
    ``moved`` column is written, True on the undulator rows of the named time
    indices.
    """
    base_time = datetime(2024, 6, 10, 4, 0)
    rows = []
    for t_idx in range(3):
        nominal = (base_time + timedelta(minutes=15 * t_idx)).isoformat()
        moved = moving_at is not None and t_idx in moving_at
        for u_idx, und in enumerate(UND_NUMBERS):
            k = 1.5 + (STEP * (u_idx - 2) if u_idx >= 3 else 0.0)
            row = (nominal, LINE.kact_pv(und), nominal, k)
            rows.append(row + (moved,) if moving_at is not None else row)
        mom = (nominal, BEAM_MOMENTUM_PV, nominal, MOMENTUM_GEV)
        rows.append(mom + (False,) if moving_at is not None else mom)
    columns = ["nominal_time", "pv", "timestamp", "value"]
    if moving_at is not None:
        columns.append("moved")
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def test_physics_gamma_and_taper() -> None:
    gamma = float(gamma_from_momentum_gev(10.0))
    assert math.isclose(gamma, 10e9 / 0.5109989500e6, rel_tol=1e-4)
    # exact dgamma agrees with the small-step Taylor form to <1%
    exact = float(dk_to_dgamma(0.01, 2.5, gamma))
    approx = gamma * (2.5 * 0.01 + 0.01**2 / 2) / (2.5**2 + 2)
    assert math.isclose(exact, approx, rel_tol=1e-2)
    assert float(taper_mev_per_fs(0.01, 2.5, gamma)) > 0


def test_gap_to_kact_hxr() -> None:
    """HXR gap (mm) -> K: monotone-decreasing, and a down-taper in gap (== an
    XLEAP up-taper in K) yields increasing K so the detector fires on the right
    sign. Magnitude is provisional; only the ordering is asserted here."""
    from taper import kact_from_gap_mm

    assert float(kact_from_gap_mm(7.0)) > float(kact_from_gap_mm(20.0)) > 0
    gaps = np.array([12.0, 11.5, 11.0, 10.5, 10.0])  # closing gap head->tail
    ks = kact_from_gap_mm(gaps)
    assert np.all(np.diff(ks) > 0)  # -> rising K -> tapered


def test_store_pivot(tmp_path) -> None:
    csv = tmp_path / "snapshots.csv"
    _synthetic_csv(csv)
    store = SnapshotStore.from_csv(csv)
    kvals = store.wide_values(KACT_PV_PATTERN)
    assert list(kvals.columns) == UND_NUMBERS  # numeric-sorted, group-labelled
    assert kvals.shape == (3, 10)
    assert store.series(BEAM_MOMENTUM_PV).iloc[0] == MOMENTUM_GEV


def test_detection_and_timeline(tmp_path) -> None:
    csv = tmp_path / "snapshots.csv"
    _synthetic_csv(csv)
    store = SnapshotStore.from_csv(csv)
    kvals = store.wide_values(KACT_PV_PATTERN)

    mask = lasing_mask(kvals, DetectionParams())
    # ramp on unds index 3..9 (7 steps) plus the fencepost base at index 2
    assert mask.iloc[0].sum() >= 7
    assert not bool(mask.iloc[0, 0])  # flat head undulator is not lasing

    points = xleap_timeline(store, line=LINE)
    assert len(points) == 3
    first = points[0]
    assert first.xleap_on and first.n_und >= 7
    assert np.isfinite(first.taper) and first.taper > 0

    frame = timeline_frame(points)
    assert list(frame.columns) == ["xleap_on", "n_und", "taper", "moving", "energy_unsteady"]
    assert frame.index.is_monotonic_increasing


def test_motion_filter(tmp_path) -> None:
    csv = tmp_path / "snapshots.csv"
    _synthetic_csv(csv, moving_at={1})  # undulators moving at the middle time
    store = SnapshotStore.from_csv(csv)

    points = xleap_timeline(store, line=LINE)
    assert len(points) == 3
    lasing, moving, quiet = points  # t0 lases, t1 moving, t2 lases

    # a moving window is force-cleared even though its K profile is a hockey stick
    assert moving.moving and not moving.xleap_on
    assert moving.n_und == 0 and math.isnan(moving.taper)

    # stationary windows are untouched
    assert lasing.xleap_on and not lasing.moving
    assert lasing.n_und >= 7 and lasing.taper > 0
    assert quiet.xleap_on and not quiet.moving

    frame = timeline_frame(points)
    assert bool(frame["moving"].loc[moving.datetime])


def test_energy_stability_gate(tmp_path) -> None:
    """A bin whose beam energy was not steady across the bin is force-cleared as
    energy-unsteady (taper NaN) while steady bins keep their taper -- the gate
    keys on energy steadiness, never on taper/energy magnitude."""
    csv = tmp_path / "snapshots.csv"
    base = datetime(2024, 6, 10, 4, 0)
    rows = []
    for t_idx in range(3):
        nominal = (base + timedelta(minutes=15 * t_idx)).isoformat()
        e_spread = 1.5 if t_idx == 1 else 1e-4  # middle bin: energy swung across the bin
        for u_idx, und in enumerate(UND_NUMBERS):
            k = 1.5 + (STEP * (u_idx - 2) if u_idx >= 3 else 0.0)
            rows.append((nominal, LINE.kact_pv(und), nominal, k, False, 0.0))
        rows.append((nominal, BEAM_MOMENTUM_PV, nominal, MOMENTUM_GEV, False, e_spread))
    cols = ["nominal_time", "pv", "timestamp", "value", "moved", "spread"]
    pd.DataFrame(rows, columns=cols).to_csv(csv, index=False)

    store = SnapshotStore.from_csv(csv)
    lasing, unsteady, quiet = xleap_timeline(store, line=LINE)  # default max=0.02

    # steady-energy bins lase with a real taper
    assert lasing.xleap_on and not lasing.energy_unsteady and lasing.taper > 0
    assert quiet.xleap_on and not quiet.energy_unsteady
    # the unsteady-energy bin: force-cleared, flagged (energy, not undulator), not dropped
    assert unsteady.energy_unsteady and not unsteady.moving
    assert not unsteady.xleap_on and unsteady.n_und == 0 and math.isnan(unsteady.taper)


def test_energy_gate_off_for_legacy_data(tmp_path) -> None:
    """CSV with no ``spread`` column (pre-gate data) => spread NaN => gate never
    fires, so results match the old behaviour."""
    csv = tmp_path / "snapshots.csv"
    _synthetic_csv(csv)  # legacy schema: no moved, no spread
    store = SnapshotStore.from_csv(csv)
    assert all(not p.energy_unsteady for p in xleap_timeline(store, line=LINE))


def test_snapshots_spread_helpers() -> None:
    from snapshots.archive import _fractional_spread, _spread_by_bucket, window_epoch
    from snapshots.config import RunConfig

    assert _fractional_spread([]) == 0.0
    assert _fractional_spread([8.0]) == 0.0  # single sample => no spread
    assert _fractional_spread([8.0, 8.0]) == 0.0
    assert math.isclose(_fractional_spread([2.0, 16.0, 8.0]), (16.0 - 2.0) / 8.0)  # median 8

    # full-bin spread uses every sample, unlike the snapshot_delta_s motion window
    cfg = RunConfig(from_time="2024-06-10T04:00:00Z", bin_seconds=900, snapshot_delta_s=5.0)
    t0 = window_epoch(cfg.from_time)
    secs = [t0 + 1, t0 + 400, t0 + 800]  # samples spread across the whole bin
    vals = [8.0, 2.0, 16.0]
    sp = _spread_by_bucket(secs, vals, cfg)
    assert math.isclose(sp[t0], (16.0 - 2.0) / 8.0)


def test_snapshots_motion_helpers() -> None:
    from snapshots.archive import _motion_by_bucket, _window_moved, window_epoch
    from snapshots.config import RunConfig

    assert not _window_moved([2.5], 1e-3)  # single sample => stationary
    assert not _window_moved([2.5, 2.5001], 1e-3)  # < 0.1% spread => thermal
    assert _window_moved([2.5, 2.51], 1e-3)  # ~0.4% spread => moving

    cfg = RunConfig(
        from_time="2024-06-10T04:00:00Z",
        bin_seconds=900,
        snapshot_delta_s=5.0,
        motion_threshold=1e-3,
    )
    t0 = window_epoch(cfg.from_time)  # 900-aligned nominal time
    secs = [t0 + 1, t0 + 3, t0 + 900 + 2]  # two in bin t0 (moving), one in t0+900
    vals = [2.50, 2.53, 2.50]
    moved = _motion_by_bucket(secs, vals, cfg)
    assert moved[t0] is True  # spread inside the 5 s window
    assert moved.get(t0 + 900, False) is False  # lone sample => stationary
    # a sample beyond the delta window is ignored, not counted as motion
    assert _motion_by_bucket([t0 + 1, t0 + 400], [2.50, 2.90], cfg).get(t0) is False


def _run_standalone() -> int:
    import tempfile
    from pathlib import Path

    test_physics_gamma_and_taper()
    test_gap_to_kact_hxr()
    with tempfile.TemporaryDirectory() as d:
        test_store_pivot(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_detection_and_timeline(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_motion_filter(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_energy_stability_gate(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_energy_gate_off_for_legacy_data(Path(d))
    test_snapshots_spread_helpers()
    test_snapshots_motion_helpers()
    print("all taper tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
