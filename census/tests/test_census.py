"""Stdlib tests for the census join logic (no pandas / no cluster needed)."""
from __future__ import annotations

from census.archiver_census import UNSCHEDULED, tally_rows
from census.common import Event, assign_event, iso_epoch, load_events, machine_of


def test_machine_of_reads_nc_sc_prefix():
    assert machine_of("NC-TMO") == "NC"
    assert machine_of("SC-CXI") == "SC"
    assert machine_of("nc-lowercase") == "NC"
    assert machine_of("") == "unknown"
    assert machine_of(None) == "unknown"
    assert machine_of("FEEH") == "unknown"  # no NC/SC prefix


def test_iso_epoch_offset_and_z_agree():
    # Same instant expressed with an offset and as Zulu must be equal.
    assert iso_epoch("2020-01-01T00:00:00-08:00") == iso_epoch("2020-01-01T08:00:00Z")


def _events():
    # Two back-to-back shifts on a line, plus a later one leaving a gap.
    return sorted(
        [
            Event(100.0, 200.0, "sase", "NC", "HXR"),
            Event(200.0, 300.0, "xleap", "SC", "HXR"),
            Event(400.0, 500.0, "self_seed", "NC", "HXR"),
        ]
    )


def test_assign_event_boundaries_and_gaps():
    events = _events()
    assert assign_event(events, 100.0).mode == "sase"   # start is inclusive
    assert assign_event(events, 199.0).mode == "sase"
    assert assign_event(events, 200.0).mode == "xleap"  # end is exclusive -> next shift
    assert assign_event(events, 299.0).mode == "xleap"
    assert assign_event(events, 350.0) is None          # gap between shifts
    assert assign_event(events, 450.0).mode == "self_seed"
    assert assign_event(events, 50.0) is None           # before all shifts
    assert assign_event([], 123.0) is None


def test_tally_rows_counts_total_delivered_and_unscheduled():
    events = _events()
    # rows: two in sase (one lasing), one in xleap (lasing), one in the gap.
    epochs = [120.0, 150.0, 250.0, 350.0]
    delivered = [True, False, True, True]
    cells = tally_rows(epochs, delivered, events)

    assert cells[("sase", "NC")].n_total == 2
    assert cells[("sase", "NC")].n_delivered == 1
    assert cells[("xleap", "SC")].n_total == 1
    assert cells[("xleap", "SC")].n_delivered == 1
    assert cells[(UNSCHEDULED, "unknown")].n_total == 1  # the gap row


def test_load_events_skips_bad_rows(tmp_path):
    csv_path = tmp_path / "hxr_classified_all.csv"
    csv_path.write_text(
        "start,end,primary_class,calendar\n"
        "2020-01-01T00:00:00-08:00,2020-01-01T08:00:00-08:00,sase,NC-CXI\n"
        ",,xleap,SC-TMO\n"                       # missing start/end -> skipped
        "not-a-date,also-bad,sase,NC-XPP\n"      # unparseable -> skipped
    )
    events = load_events(csv_path, "HXR")
    assert len(events) == 1
    assert events[0].mode == "sase" and events[0].machine == "NC" and events[0].line == "HXR"
