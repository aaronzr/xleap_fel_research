"""Undulator-taper physics -- pure, vectorized, no I/O.

Every function accepts NumPy scalars or arrays and broadcasts, so the same code
serves one time point or a whole time series. Symbols follow the notebook's
derivation (Kim, Huang & Lindberg, *Synchrotron Radiation and FELs*, 2017):

    gamma : Lorentz factor of the beam
    K     : undulator deflection parameter
    dK    : change in K between adjacent undulators (drives the taper)

Reference relations used below::

    lambda_r = (lambda_u / 2 gamma^2) (1 + K^2 / 2)                 # resonance
    dgamma   = gamma (sqrt(1 + (K + dK/2) dK / (1 + K^2/2)) - 1)    # to hold it
    taper    = 2 gamma^2 dgamma c / (L_u (1 + K^2/2))               # dgamma / slippage
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import (
    ELECTRON_REST_ENERGY_MEV,
    SPEED_OF_LIGHT_M_PER_S,
    UNDULATOR_LENGTH_M,
)

__all__ = [
    "gamma_from_momentum_gev",
    "dk_to_dgamma",
    "slippage_time_s",
    "taper_mev_per_fs",
    "kact_from_gap_mm",
]

# --- Undulator gap -> K (HXR only) -------------------------------------------
# The hard line archives GapAct (mm), not KAct, so we reconstruct K from gap with
# the planar-hybrid Halbach peak-field form and K = 0.9337 * B0[T] * lambda_u[cm]:
#     r  = gap / lambda_u
#     B0 = a * exp(-b*r + c*r^2)      [T]
#     K  = 0.09337 * B0 * lambda_u[mm]
#
# !!! PROVISIONAL CONSTANTS !!!  Generic NdFeB-hybrid coefficients with the LCLS
# HXR period -- enough to get the taper *sign* right (K is monotone-decreasing in
# gap, so an XLEAP up-taper in K = a down-taper in gap), but the absolute taper
# magnitude (MeV/fs) is NOT trustworthy until replaced.
# TODO(undulator group / Aaron Reed): drop in the measured HXR K(gap) calibration,
# or fit a,b,c from an archived window where BOTH KAct and GapAct exist.
HXR_UNDULATOR_PERIOD_MM: float = 26.0
_HALBACH_A: float = 3.694
_HALBACH_B: float = 5.068
_HALBACH_C: float = 1.520


def kact_from_gap_mm(gap_mm: ArrayLike) -> NDArray[np.float64]:
    """Undulator K reconstructed from gap (mm) via the Halbach peak-field form.

    Vectorized/broadcasting like the rest of this module; monotone-decreasing in
    gap. Constants are PROVISIONAL (sign trustworthy, magnitude not) -- see the
    module note above.
    """
    g = np.asarray(gap_mm, dtype=np.float64)
    r = g / HXR_UNDULATOR_PERIOD_MM
    b0 = _HALBACH_A * np.exp(-_HALBACH_B * r + _HALBACH_C * r**2)
    return 0.09337 * b0 * HXR_UNDULATOR_PERIOD_MM


def gamma_from_momentum_gev(momentum_gev_c: ArrayLike) -> NDArray[np.float64]:
    """Lorentz factor from beam momentum in GeV/c.

    The active line's dump-magnet PV (``BEND:DMPS:400:BACT`` on SXR,
    ``BEND:DMPH:400:BACT`` on HXR) reports the momentum the dump is tuned to bend,
    in GeV/c. Exactly, ``gamma = E / m c^2 = sqrt((pc)^2 + (m c^2)^2) / m c^2``;
    since ``pc >> m c^2`` this is ~``pc / m c^2`` but we keep the exact form.
    """
    pc_mev = np.asarray(momentum_gev_c, dtype=np.float64) * 1.0e3
    return np.sqrt((pc_mev / ELECTRON_REST_ENERGY_MEV) ** 2 + 1.0)


def dk_to_dgamma(
    dK: ArrayLike, K: ArrayLike, gamma: ArrayLike
) -> NDArray[np.float64]:
    """Energy change (in units of gamma) needed to keep resonance across ``dK``.

    Solves ``lambda_r(gamma, K) = lambda_r(gamma + dgamma, K + dK)`` for dgamma.
    """
    dK = np.asarray(dK, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    return gamma * (np.sqrt(1.0 + ((K + dK / 2.0) * dK) / (1.0 + K**2 / 2.0)) - 1.0)


def slippage_time_s(K: ArrayLike, gamma: ArrayLike) -> NDArray[np.float64]:
    """Time for radiation to slip one undulator's worth ahead of the beam.

    One undulator holds ``N_u = L_u / lambda_u`` periods, each slipping the beam
    by one ``lambda_r``; the total slippage ``N_u lambda_r`` in units of time is
    ``L_u (1 + K^2/2) / (2 gamma^2 c)`` -- and ``lambda_u`` cancels.
    """
    K = np.asarray(K, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    return UNDULATOR_LENGTH_M * (1.0 + K**2 / 2.0) / (2.0 * gamma**2 * SPEED_OF_LIGHT_M_PER_S)


def taper_mev_per_fs(
    dK: ArrayLike, K: ArrayLike, gamma: ArrayLike
) -> NDArray[np.float64]:
    """Undulator taper: resonant-energy gain per slippage length, in MeV/fs.

    ``taper = dgamma / slippage_time``, with ``dgamma`` converted to MeV
    (x ``m c^2``) and ``slippage_time`` to fs (x 1e15).
    """
    dgamma = dk_to_dgamma(dK, K, gamma)
    slippage_fs = slippage_time_s(K, gamma) * 1.0e15
    return dgamma * ELECTRON_REST_ENERGY_MEV / slippage_fs
