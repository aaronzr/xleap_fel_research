"""Tests for the ``server`` package.

Exercises the router end-to-end against a *synthetic archive* (an injected
``fetch`` stand-in), so no live MEME connection is needed. Runs standalone
(``python tests/test_server.py``) or under pytest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Repo root importable so `server` / `snapshots` / `taper` resolve either way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import Backend, Router  # noqa: E402
from snapshots.archive import iso_utc, window_epoch  # noqa: E402
from taper import DEFAULT_LINE as LINE  # noqa: E402

MOM_PV = LINE.momentum_pv
UND_NUMBERS = [str(1450 + 100 * i) for i in range(10)]  # 1450..2350
UND_PVS = [LINE.kact_pv(u) for u in UND_NUMBERS]
MOMENTUM_GEV = 4.0
STEP = 0.05  # per-undulator K rise inside the planted group (>> 4*rho)

START = "2024-06-10T04:00:00Z"
END = "2024-06-10T04:45:00Z"  # 3 nominal times at interval=900


def make_fake(moving_at: set[int] | None = None):
    """A ``snapshots.fetch_pv`` stand-in over a planted hockey-stick K matrix.

    Returns ``(fetch, calls)``; ``calls['n']`` counts archive round-trips so a
    test can assert the cache prevented a second fetch. ``moving_at`` names the
    time-index rows whose undulators are flagged as moving.
    """
    moving_at = moving_at or set()
    calls = {"n": 0}

    def fetch(pv, cfg):
        calls["n"] += 1
        t0, t_end = window_epoch(cfg.from_time), window_epoch(cfg.to_time)
        rows = []
        idx, t = 0, t0
        while t < t_end:
            nominal = iso_utc(t)
            if pv == MOM_PV:
                rows.append((nominal, pv, nominal, MOMENTUM_GEV, False, 0.0))
            elif pv in UND_PVS:
                u_idx = UND_PVS.index(pv)
                k = 1.5 + (STEP * (u_idx - 2) if u_idx >= 3 else 0.0)
                rows.append((nominal, pv, nominal, k, idx in moving_at, 0.0))
            else:
                return pv, [], "empty"
            idx, t = idx + 1, t + cfg.bin_seconds
        return pv, rows, "ok"

    return fetch, calls


def _router(tmp_path, moving_at=None) -> tuple[Router, dict]:
    fetch, calls = make_fake(moving_at)
    backend = Backend(line=LINE, fetch=fetch, pvs=[MOM_PV, *UND_PVS], jobs=2)
    return Router(backend=backend, cache_dir=tmp_path), calls


def _q(**extra):
    query = {"start": [START], "end": [END]}
    query.update({k: [str(v)] for k, v in extra.items()})
    return query


def test_health_and_listing(tmp_path) -> None:
    router, _ = _router(tmp_path)
    status, body = router.dispatch("GET", "/health", {})
    assert status == 200 and body["status"] == "ok"

    status, body = router.dispatch("GET", "/pvs", {})
    assert status == 200
    assert MOM_PV in body["pvs"] and "taper" in body["quantities"]


def test_pv_endpoint_and_cache(tmp_path) -> None:
    router, calls = _router(tmp_path)
    status, body = router.dispatch("GET", f"/pv/{MOM_PV}", _q())
    assert status == 200
    assert body["pv"] == MOM_PV and body["window"] == 5.0 and body["interval"] == 900
    assert body["count"] == 3
    assert all(v["value"] == MOMENTUM_GEV for v in body["values"])
    assert isinstance(body["values"][0]["value"], float)  # coerced from CSV text

    first = calls["n"]
    # second identical request must be served from cache -> no new archive call
    status, body2 = router.dispatch("GET", f"/pv/{MOM_PV}", _q())
    assert status == 200 and calls["n"] == first
    assert body2["values"] == body["values"]


def test_taper_and_n_und(tmp_path) -> None:
    router, _ = _router(tmp_path)
    status, taper = router.dispatch("GET", "/taper", _q())
    assert status == 200 and taper["quantity"] == "taper"
    assert taper["count"] == 3
    assert all(v["value"] is not None and v["value"] > 0 for v in taper["values"])

    status, n_und = router.dispatch("GET", "/n_und", _q())
    assert status == 200 and n_und["quantity"] == "n_und"
    assert all(v["value"] >= 7 for v in n_und["values"])  # ramp of 7 + fencepost


def test_derived_cache_shared_across_quantities(tmp_path) -> None:
    """taper and n_und read the same cached derived timeline -> one pull."""
    router, calls = _router(tmp_path)
    router.dispatch("GET", "/taper", _q())
    after_taper = calls["n"]
    assert after_taper > 0  # all PVs pulled once
    router.dispatch("GET", "/n_und", _q())
    assert calls["n"] == after_taper  # served from the derived cache, no refetch


def test_pull_all_filters_motion_and_caches(tmp_path) -> None:
    router, _ = _router(tmp_path, moving_at={1})  # undulators moving at middle time
    status, body = router.dispatch("GET", "/pull_all", _q())
    assert status == 200 and body["count"] == 3
    lasing, moving, quiet = body["values"]

    # the moving time point is skipped for taper / lasing-group calc but kept visible
    assert moving["moving"] and not moving["xleap_on"]
    assert moving["n_und"] == 0 and moving["taper"] is None
    assert lasing["xleap_on"] and lasing["n_und"] >= 7 and lasing["taper"] > 0
    assert quiet["xleap_on"] and not quiet["moving"]

    # results were cached to CSV, as the notebook/scripts do
    assert (tmp_path / "derived_cache.csv").exists()
    assert (tmp_path / "derived_cache.csv.coverage.json").exists()


def test_pull_all_then_taper_uses_cache(tmp_path) -> None:
    router, calls = _router(tmp_path)
    router.dispatch("GET", "/pull_all", _q())
    after_pull = calls["n"]
    # a taper request fully inside the pulled window must not hit the archive
    router.dispatch("GET", "/taper", _q())
    assert calls["n"] == after_pull


def test_custom_window_and_interval_are_separate_cache_buckets(tmp_path) -> None:
    router, calls = _router(tmp_path)
    router.dispatch("GET", f"/pv/{MOM_PV}", _q())
    after_default = calls["n"]
    # a different interval is a different key -> a fresh fetch, not a false hit
    router.dispatch("GET", f"/pv/{MOM_PV}", _q(interval=300))
    assert calls["n"] > after_default


def test_bad_requests(tmp_path) -> None:
    router, _ = _router(tmp_path)
    status, body = router.dispatch("GET", "/taper", {"start": [START]})  # no end
    assert status == 400 and "error" in body

    status, _ = router.dispatch("GET", "/nope", _q())
    assert status == 404

    status, _ = router.dispatch("POST", "/taper", _q())
    assert status == 405

    status, body = router.dispatch("GET", "/taper", {"start": ["not-a-date"], "end": [END]})
    assert status == 400


def test_response_is_json_serializable(tmp_path) -> None:
    router, _ = _router(tmp_path)
    _, body = router.dispatch("GET", "/pull_all", _q())
    json.dumps(body)  # must not raise


def _run_standalone() -> int:
    import tempfile

    tests = [
        test_health_and_listing,
        test_pv_endpoint_and_cache,
        test_taper_and_n_und,
        test_derived_cache_shared_across_quantities,
        test_pull_all_filters_motion_and_caches,
        test_pull_all_then_taper_uses_cache,
        test_custom_window_and_interval_are_separate_cache_buckets,
        test_bad_requests,
        test_response_is_json_serializable,
    ]
    for test in tests:
        with tempfile.TemporaryDirectory() as d:
            test(Path(d))
    print("all server tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
