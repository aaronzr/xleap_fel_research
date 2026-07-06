"""Data census: how much data exists per (mode x line x machine) cell.

Two tiers, sharing the interval-join and NC/SC rules in :mod:`census.common`:

* :mod:`census.calendar_census` (Tier A) -- SCHEDULED shift-hours/events from the
  classified calendar CSVs. Stdlib-only, runs anywhere.
* :mod:`census.archiver_census` (Tier B) -- DELIVERED 1 Hz archiver rows per cell,
  by joining those same events against the S3DF ``.pkl`` set. Needs pandas + the
  cluster (``iana``); HXR only until an SXR archiver pull exists.
"""
