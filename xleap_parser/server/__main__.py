"""CLI entrypoint: ``python -m server``.

Owns argv parsing and process wiring only; the routing/analysis live in
:mod:`server.app` / :mod:`server.compute`. Run from ``xleap_parser/`` so the
default ``pvs_<line>.csv`` and the ``snapshots`` / ``taper`` packages resolve::

    cd xleap_parser
    python -m server --host 0.0.0.0 --port 8000 --cache-dir cache
"""
from __future__ import annotations

import argparse

from .app import serve
from .compute import Backend


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m server", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    p.add_argument("--cache-dir", default="cache", help="cache directory (default: ./cache)")
    p.add_argument("--pvs", default=None,
                   help="PV-list CSV to serve (default: the active line's pvs_<line>.csv)")
    p.add_argument("--jobs", type=int, default=8, help="concurrent archive fetches (default: 8)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from snapshots.__main__ import read_pvs

    pvs = read_pvs(args.pvs) if args.pvs else []
    backend = Backend(pvs=pvs, jobs=args.jobs)
    serve(host=args.host, port=args.port, cache_dir=args.cache_dir, backend=backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
