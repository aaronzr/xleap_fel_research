"""Request routing and a stdlib ``http.server`` handler.

The "main" layer: it owns URL parsing, the cache-before-MEME decision, JSON
shaping, and HTTP status codes -- but delegates every number to
:class:`server.compute.Backend`. Routing is factored into
:meth:`Router.dispatch`, a pure ``(method, path, params) -> (status, body)``
function, so the endpoints can be tested without binding a socket.

Endpoints (all ``GET``)::

    /health                 liveness probe
    /pvs                    list the PVs / quantities this server serves
    /pv/<PV>                timestamped values for one PV
    /taper                  derived taper (MeV/fs) per nominal time
    /n_und                  derived number of lasing undulators per nominal time
    /pull_all               pull all PVs, compute+cache the whole derived timeline

Query params: ``start`` & ``end`` (ISO8601 UTC, required), ``window`` (snapshot
motion window in seconds, default 5), ``interval`` (nominal-grid spacing in
seconds, default 900 = 15 min).
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .cache import TimeSeriesCache, cache_key, parse_dt
from .compute import DERIVED_COLUMNS, PV_COLUMNS, ArchiveError, Backend

__all__ = ["Router", "make_handler", "serve", "DEFAULT_WINDOW_S", "DEFAULT_INTERVAL_S"]

DEFAULT_WINDOW_S = 5.0
DEFAULT_INTERVAL_S = 900  # 15 minutes

# How each cached (string) cell is coerced back to a typed JSON value.
_BOOL_FIELDS = {"moved", "xleap_on", "moving", "energy_unsteady"}
_INT_FIELDS = {"n_und"}
_FLOAT_FIELDS = {"value", "spread", "taper"}


class RequestError(Exception):
    """A bad request: carries the HTTP status to return (default 400)."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _as_bool(text: str) -> bool:
    return str(text).strip().lower() in ("true", "1")


def _coerce(row: dict[str, Any]) -> dict[str, Any]:
    """Type a cached row's string cells for JSON (bools, ints, floats)."""
    out: dict[str, Any] = {}
    for field, value in row.items():
        if field in _BOOL_FIELDS:
            out[field] = _as_bool(value)
        elif field in _INT_FIELDS:
            out[field] = int(float(value)) if value not in ("", None) else None
        elif field in _FLOAT_FIELDS:
            if value in ("", None):
                out[field] = None
            else:
                number = float(value)
                # NaN (an uncomputable taper) is not valid JSON: surface it as null.
                out[field] = None if number != number else number
        else:
            out[field] = value
    return out


@dataclass
class Params:
    """A validated request window plus the snapshot/interval knobs."""

    start: datetime
    end: datetime
    window: float
    interval: int


@dataclass
class Router:
    """Wire the compute backend to the two CSV caches and route requests."""

    backend: Backend
    cache_dir: Path

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self._pv_cache = TimeSeriesCache(self.cache_dir / "pv_cache.csv", PV_COLUMNS)
        self._derived_cache = TimeSeriesCache(
            self.cache_dir / "derived_cache.csv", DERIVED_COLUMNS
        )

    # --- param parsing -------------------------------------------------------
    @staticmethod
    def _parse_params(query: dict[str, list[str]]) -> Params:
        def one(name: str) -> str | None:
            values = query.get(name)
            return values[0] if values else None

        start_raw, end_raw = one("start"), one("end")
        if not start_raw or not end_raw:
            raise RequestError("both 'start' and 'end' query params are required")
        try:
            start, end = parse_dt(start_raw), parse_dt(end_raw)
        except ValueError as exc:
            raise RequestError(f"could not parse start/end as ISO8601: {exc}")
        if end <= start:
            raise RequestError("'end' must be after 'start'")
        try:
            window = float(one("window") or DEFAULT_WINDOW_S)
            interval = int(one("interval") or DEFAULT_INTERVAL_S)
        except ValueError as exc:
            raise RequestError(f"window/interval must be numeric: {exc}")
        if window < 0 or interval <= 0:
            raise RequestError("window must be >= 0 and interval must be > 0")
        return Params(start=start, end=end, window=window, interval=interval)

    # --- cache-before-MEME core ---------------------------------------------
    def _pv_rows(self, pv: str, p: Params) -> list[dict[str, Any]]:
        key = cache_key(pv, window=p.window, interval=p.interval)
        if not self._pv_cache.covered(key, p.start, p.end):
            rows = self.backend.pv_series(pv, p.start, p.end, p.interval, p.window)
            self._pv_cache.put(key, p.start, p.end, rows)
        return [_coerce(r) for r in self._pv_cache.get(key, p.start, p.end)]

    def _derived_rows(self, p: Params) -> list[dict[str, Any]]:
        key = cache_key("derived", window=p.window, interval=p.interval)
        if not self._derived_cache.covered(key, p.start, p.end):
            rows = self.backend.derived_rows(p.start, p.end, p.interval, p.window)
            self._derived_cache.put(key, p.start, p.end, rows)
        return [_coerce(r) for r in self._derived_cache.get(key, p.start, p.end)]

    # --- dispatch ------------------------------------------------------------
    def dispatch(
        self, method: str, path: str, query: dict[str, list[str]]
    ) -> tuple[int, Any]:
        """Route one request to a JSON body + status. Never raises."""
        try:
            if method != "GET":
                raise RequestError("only GET is supported", status=405)
            return 200, self._route(path, query)
        except RequestError as exc:
            return exc.status, {"error": str(exc)}
        except ArchiveError as exc:
            return 502, {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - surface any bug as a 500, not a crash
            traceback.print_exc()  # full trace to stderr to aid debugging
            return 500, {"error": f"{type(exc).__name__}: {exc}"}

    def _route(self, path: str, query: dict[str, list[str]]) -> Any:
        path = path.rstrip("/") or "/"
        if path in ("/", "/health"):
            return {"status": "ok"}
        if path == "/pvs":
            return {
                "line": self.backend.line.name,
                "pvs": self.backend.pvs,
                "quantities": ["taper", "n_und"],
                "endpoints": ["/pv/<PV>", "/taper", "/n_und", "/pull_all"],
            }

        p = self._parse_params(query)
        meta = {
            "start": query["start"][0],
            "end": query["end"][0],
            "window": p.window,
            "interval": p.interval,
        }

        if path.startswith("/pv/"):
            pv = unquote(path[len("/pv/"):])
            if not pv:
                raise RequestError("missing PV name in /pv/<PV>")
            values = self._pv_rows(pv, p)
            return {"pv": pv, **meta, "count": len(values), "values": values}

        if path == "/taper":
            rows = self._derived_rows(p)
            values = [{"timestamp": r["timestamp"], "value": r["taper"]} for r in rows]
            return {"quantity": "taper", "units": "MeV/fs", **meta,
                    "count": len(values), "values": values}

        if path == "/n_und":
            rows = self._derived_rows(p)
            values = [{"timestamp": r["timestamp"], "value": r["n_und"]} for r in rows]
            return {"quantity": "n_und", "units": "undulators", **meta,
                    "count": len(values), "values": values}

        if path == "/pull_all":
            rows = self._derived_rows(p)
            return {**meta, "line": self.backend.line.name,
                    "pvs": self.backend.pvs, "count": len(rows), "values": rows}

        raise RequestError(f"no such endpoint: {path}", status=404)


# --- HTTP glue ---------------------------------------------------------------
def make_handler(router: Router) -> type[BaseHTTPRequestHandler]:
    """A ``BaseHTTPRequestHandler`` subclass bound to ``router``."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "xleap-parser/1.0"

        def do_GET(self) -> None:  # noqa: N802 - name mandated by BaseHTTPRequestHandler
            parts = urlsplit(self.path)
            status, body = router.dispatch("GET", parts.path, parse_qs(parts.query))
            self._send(status, body)

        def _send(self, status: int, body: Any) -> None:
            payload = json.dumps(body, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:  # keep test output quiet
            pass

    return Handler


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    cache_dir: str | Path = "cache",
    backend: Backend | None = None,
) -> None:
    """Run the blocking threaded HTTP server (used by ``python -m server``)."""
    router = Router(backend=backend or Backend(), cache_dir=Path(cache_dir))
    httpd = ThreadingHTTPServer((host, port), make_handler(router))
    print(f"xleap_parser server on http://{host}:{port}  (cache: {router.cache_dir})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
