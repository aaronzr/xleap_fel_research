"""HTTP API server over the archived-snapshot analysis.

Exposes the same numbers the notebook and ``python -m snapshots`` +
:mod:`taper` produce, but on demand and per quantity:

    * one endpoint per PV        -- ``GET /pv/<PV>``
    * one endpoint per quantity  -- ``GET /taper``, ``GET /n_und``
    * a bulk endpoint            -- ``GET /pull_all``

Every endpoint takes ``start``/``end`` (ISO8601 UTC) plus an optional snapshot
``window`` (default 5 s) and ``interval`` (default 900 s = 15 min), and answers
with a list of timestamped values. Results are cached to CSV (see
:mod:`server.cache`); a request checks the cache before it ever calls MEME.

Layers mirror the ``snapshots`` / ``taper`` packages' library/main split:
    cache    -- CSV-backed time-series cache with coverage tracking
    compute  -- wrap snapshots.fetch_pv + taper.xleap_timeline (pure, injectable)
    app      -- request routing + a stdlib http.server handler
    __main__ -- ``python -m server`` CLI entrypoint
"""
from __future__ import annotations

from .app import Router, make_handler, serve
from .cache import TimeSeriesCache
from .compute import Backend

__all__ = ["Router", "make_handler", "serve", "TimeSeriesCache", "Backend"]
