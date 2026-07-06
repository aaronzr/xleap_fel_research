#!/usr/bin/env python3
"""Probe which archive windows actually have data for the XLEAP hard-line PVs.

Fetches a couple of key PVs across several years so you can find a window where
the USEG:UNDH undulators existed (they are LCLS-II-era and were not archived in
2018). Run from the repo root:

    python xleap_parser/scripts/probe_archive.py
"""
from datetime import datetime, timezone

import meme.archive as archive

PVS = ["USEG:UNDH:1450:KAct", "BEND:DMPH:400:BACT"]
YEARS = [2018, 2020, 2022, 2024]


def probe(pv: str, year: int) -> str:
    """Return a short OK/ERR summary for one PV over a one-day window in March."""
    query = f"firstSample_900({pv})"
    start = datetime(year, 3, 1, tzinfo=timezone.utc)
    end = datetime(year, 3, 2, tzinfo=timezone.utc)
    try:
        data = archive.get(
            query, from_time=start, to_time=end, protocol="PVA", timeout=60
        )
        return f"OK   {str(data)[:80]}"
    except Exception as exc:  # noqa: BLE001 - we just want to see which windows work
        return f"ERR  {exc}"


def main() -> None:
    for pv in PVS:
        for year in YEARS:
            print(f"{pv:24} {year}: {probe(pv, year)}")


if __name__ == "__main__":
    main()

