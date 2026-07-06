"""Detect XLEAP and quantify undulator taper from archived K snapshots.

Consumes the long-form ``snapshots.csv`` produced by ``python -m snapshots`` --
no sparklines / wide-SQLite dependency. Typical use::

    from taper import SnapshotStore, xleap_timeline

    store = SnapshotStore.from_csv("snapshots.csv")
    for point in xleap_timeline(store):
        print(point.datetime, point.xleap_on, point.n_und, point.taper)

Layers (mirroring the ``snapshots`` package's library/main split):
    constants  -- physical constants + PV naming
    physics    -- pure, vectorized taper math
    detect     -- lasing-group detection (binary erosion/dilation)
    store      -- long snapshots.csv -> wide pandas frames
    service    -- per-time XLEAP timeline assembly
"""
from __future__ import annotations

from .constants import DEFAULT_LINE, HXR, SXR, BEAM_MOMENTUM_PV, KACT_PV_PATTERN, Beamline
from .detect import DetectionParams, first_group, lasing_kvals, lasing_mask
from .physics import (
    dk_to_dgamma,
    gamma_from_momentum_gev,
    kact_from_gap_mm,
    slippage_time_s,
    taper_mev_per_fs,
)
from .service import XleapPoint, timeline_frame, xleap_timeline
from .store import SnapshotStore

__all__ = [
    "Beamline",
    "DEFAULT_LINE",
    "HXR",
    "SXR",
    "BEAM_MOMENTUM_PV",
    "KACT_PV_PATTERN",
    "DetectionParams",
    "first_group",
    "lasing_kvals",
    "lasing_mask",
    "dk_to_dgamma",
    "gamma_from_momentum_gev",
    "kact_from_gap_mm",
    "slippage_time_s",
    "taper_mev_per_fs",
    "SnapshotStore",
    "XleapPoint",
    "timeline_frame",
    "xleap_timeline",
]
