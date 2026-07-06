"""Assemble the XLEAP timeline -- the notebook's ``get_xleap_data``, as an API.

For each nominal time in the store this reports whether XLEAP is on, which
undulators lase, and the taper of the first (most-upstream) lasing group. The
heavy lifting lives in :mod:`taper.detect` and :mod:`taper.physics`; this module
only wires the wide K matrix and the gamma series together per time point.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .constants import DEFAULT_LINE, Beamline
from .detect import DetectionParams, first_group, lasing_kvals
from .physics import gamma_from_momentum_gev, kact_from_gap_mm, taper_mev_per_fs
from .store import SnapshotStore

__all__ = [
    "XleapPoint",
    "xleap_timeline",
    "timeline_frame",
    "DEFAULT_ENERGY_SPREAD_MAX",
]

# Max fractional intra-bin spread of the beam-energy readback before the bin's
# energy is deemed not-steady (a transition / dump-magnet standardization), which
# makes gamma -- and therefore the taper -- uncomputable there. Provisional value;
# tune with the physics/controls group. Normal beam-energy jitter is <<1%; the
# observed standardization excursions spread ~170%, so the gate has wide margin.
DEFAULT_ENERGY_SPREAD_MAX: float = 0.02


@dataclass(frozen=True)
class XleapPoint:
    """One nominal time's XLEAP verdict.

    ``taper`` is the first lasing group's taper in MeV/fs (NaN when off or when
    gamma is unavailable). ``k_lasing`` maps undulator number -> K over every
    lasing undulator at this time. Two independent quality flags force-clear a
    point (no taper, no XLEAP) while keeping it visible rather than dropping it:
    ``moving`` -- an undulator was being slewed at the snapshot instant; and
    ``energy_unsteady`` -- the beam-energy readback was not steady across the bin
    (a transition/standardization), so its gamma cannot be trusted.
    """

    datetime: datetime
    xleap_on: bool
    k_lasing: dict[str, float]
    n_und: int
    taper: float
    moving: bool = False
    energy_unsteady: bool = False

    def as_dict(self) -> dict[str, Any]:
        """The notebook's documented data object (``K_lasing`` spelling kept)."""
        return {
            "datetime": self.datetime,
            "xleap_on": self.xleap_on,
            "K_lasing": self.k_lasing,
            "n_und": self.n_und,
            "taper": self.taper,
            "moving": self.moving,
            "energy_unsteady": self.energy_unsteady,
        }


def _group_taper(group: pd.Series, gamma: float) -> float:
    """Median-dK taper of one lasing group at beam energy ``gamma``."""
    if len(group) < 2 or not np.isfinite(gamma):
        return float("nan")
    dK = group.diff().dropna().median()
    K = float(group.iloc[0])
    if not np.isfinite(dK):
        return float("nan")
    return float(taper_mev_per_fs(dK, K, gamma))


def xleap_timeline(
    store: SnapshotStore,
    *,
    line: Beamline = DEFAULT_LINE,
    params: DetectionParams = DetectionParams(),
    start: datetime | None = None,
    end: datetime | None = None,
    energy_spread_max: float = DEFAULT_ENERGY_SPREAD_MAX,
) -> list[XleapPoint]:
    """XLEAP verdict per nominal time in ``store`` (optionally clipped to a window).

    ``line`` selects the undulator line's PV naming (``HXR`` or ``SXR``).
    ``energy_spread_max`` is the max fractional intra-bin spread of the beam-energy
    readback tolerated before the bin is force-cleared as energy-unsteady (the
    taper's gamma is untrustworthy there). The gate is skipped for bins whose
    spread is unknown (legacy data with no ``spread`` column -> NaN).
    """
    kvals = store.wide_values(line.kact_pattern)
    if line.seg_attr == "GapAct":
        # HXR archives undulator gap (mm), not K; convert so the K-based detector
        # and taper physics stay valid. SXR is native KAct -> identity (skipped).
        kvals = pd.DataFrame(
            kact_from_gap_mm(kvals.to_numpy()),
            index=kvals.index,
            columns=kvals.columns,
        )
    if start is not None:
        kvals = kvals.loc[kvals.index >= pd.Timestamp(start)]
    if end is not None:
        kvals = kvals.loc[kvals.index <= pd.Timestamp(end)]

    momentum = store.series(line.momentum_pv).reindex(kvals.index)
    gamma = pd.Series(gamma_from_momentum_gev(momentum.to_numpy()), index=kvals.index)
    energy_spread = store.spread_series(line.momentum_pv).reindex(kvals.index)

    masked = lasing_kvals(kvals, params)
    moved = store.wide_moved(line.kact_pattern)
    moving_any = moved.any(axis=1).reindex(kvals.index, fill_value=False)

    def _cleared(time, *, moving: bool, energy_unsteady: bool) -> XleapPoint:
        """A force-cleared point: uncomputable taper, kept visible and flagged."""
        return XleapPoint(
            datetime=time.to_pydatetime(),
            xleap_on=False,
            k_lasing={},
            n_und=0,
            taper=float("nan"),
            moving=moving,
            energy_unsteady=energy_unsteady,
        )

    points: list[XleapPoint] = []
    for time, row in masked.iterrows():
        if bool(moving_any.loc[time]):
            # Undulators were moving in this window: don't trust a taper here
            # (Aaron). Force-clear but keep the point flagged, not dropped.
            points.append(_cleared(time, moving=True, energy_unsteady=False))
            continue
        spread = energy_spread.loc[time]
        if np.isfinite(spread) and spread > energy_spread_max:
            # Beam energy was not steady across the bin (a transition or dump-magnet
            # standardization): gamma -- and thus taper -- is uncomputable here.
            points.append(_cleared(time, moving=False, energy_unsteady=True))
            continue
        lasing = row.dropna()
        k_lasing = {str(und): float(k) for und, k in lasing.items()}
        taper = _group_taper(first_group(row), float(gamma.loc[time]))
        points.append(
            XleapPoint(
                datetime=time.to_pydatetime(),
                xleap_on=bool(k_lasing),
                k_lasing=k_lasing,
                n_und=len(k_lasing),
                taper=taper,
                moving=False,
                energy_unsteady=False,
            )
        )
    return points


def timeline_frame(points: list[XleapPoint]) -> pd.DataFrame:
    """Tabulate a timeline for plotting: index = datetime, sorted ascending."""
    frame = pd.DataFrame(
        {
            "xleap_on": [p.xleap_on for p in points],
            "n_und": [p.n_und for p in points],
            "taper": [p.taper for p in points],
            "moving": [p.moving for p in points],
            "energy_unsteady": [p.energy_unsteady for p in points],
        },
        index=pd.DatetimeIndex([p.datetime for p in points], name="datetime"),
    )
    return frame.sort_index()
