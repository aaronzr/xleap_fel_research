"""THE undulator-line switch -- the single knob to flip (HXR <-> SXR).

Both sides of the tool read ``DEFAULT_LINE`` from here, so the fetched PV list and
the analysis' PV parsing can never disagree:

    * fetch    -- ``snapshots.constants.DEFAULT_PVS_CSV`` = ``DEFAULT_LINE.pvs_csv``
    * analysis -- ``taper.xleap_timeline(..., line=DEFAULT_LINE)`` (notebook default)

To switch lines: set ``DEFAULT_LINE`` at the bottom to ``HXR`` or ``SXR``, then
re-fetch and re-run. Nothing else needs editing.

Kept dependency-free (stdlib only) on purpose, so the light ``snapshots`` fetcher
can import it without pulling in numpy/pandas/scipy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# PV-list CSVs live in ``config/`` next to this module. Anchoring to ``__file__``
# (not the process CWD) is what lets the fetcher find them whether it is run from
# the repo root, from ``xleap_parser/``, or via ``scripts/pull_sxr.py`` -- one
# source of truth for the path, resolved deterministically.
_CONFIG_DIR: Final[Path] = Path(__file__).resolve().parent / "config"


@dataclass(frozen=True)
class Beamline:
    """Line-specific PV naming for one undulator line (HXR vs SXR).

    ``kact_pattern``'s first capture group is the undulator number. ``momentum_pv``
    is the dump-magnet momentum in GeV/c that gives gamma (DMPH = hard line,
    DMPS = soft line).

    ``seg_attr`` names the per-segment quantity the pattern actually matches:
    ``"KAct"`` on SXR (K is archived, used directly), but ``"GapAct"`` on HXR --
    the hard line does NOT archive ``KAct``, only undulator gap (mm). The pipeline
    converts gap -> K downstream (``physics.kact_from_gap_mm``) so the K-based
    detector and taper physics stay valid; SXR (``"KAct"``) skips that conversion.
    """

    name: str
    kact_pattern: re.Pattern[str]
    momentum_pv: str
    seg_attr: str = "KAct"

    @property
    def pvs_csv(self) -> str:
        """Absolute path to this line's PV list (``config/pvs_hxr.csv`` / ``pvs_sxr.csv``).

        Anchored to the module via :data:`_CONFIG_DIR`, so it resolves the same no
        matter what the process CWD is.
        """
        return str(_CONFIG_DIR / f"pvs_{self.name.lower()}.csv")

    def kact_pv(self, undulator: object) -> str:
        """Concrete per-segment PV name for one undulator (inverse of ``kact_pattern``).

        Derived from ``kact_pattern`` so the line's device string has a single
        source of truth: ``USEG:UNDS:(\\d+):KAct`` -> ``USEG:UNDS:2150:KAct`` (SXR),
        ``USEG:UNDH:(\\d+):GapAct`` -> ``USEG:UNDH:1450:GapAct`` (HXR).
        """
        return self.kact_pattern.pattern.replace(r"(\d+)", str(undulator))


HXR: Final[Beamline] = Beamline(
    name="HXR",
    # KAct is NOT archived on the hard line -- match GapAct (mm) and convert to K
    # downstream (see ``seg_attr`` / ``physics.kact_from_gap_mm``).
    kact_pattern=re.compile(r"USEG:UNDH:(\d+):GapAct"),
    momentum_pv="BEND:DMPH:400:BACT",
    seg_attr="GapAct",
)
SXR: Final[Beamline] = Beamline(
    name="SXR",
    kact_pattern=re.compile(r"USEG:UNDS:(\d+):KAct"),
    momentum_pv="BEND:DMPS:400:BACT",
    seg_attr="KAct",
)

# ============================================================================
#  THE ONE SWITCH  --  set to HXR or SXR, then re-fetch and re-run. Nothing else.
# ============================================================================
DEFAULT_LINE: Final[Beamline] = SXR
