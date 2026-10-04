# Gate tests

"""test_gate.py
Tests for the synchronization gate behavior (early release, timeout, late arrivals, duplicate handling).
"""

import time
import pathlib

from gate.sync_gate import SyncGate

# Helper to build a minimal result dict matching the common interface
def make_result(clip_id, modality, status="success", features=None, predictions=None, quality_metrics=None):
    return {
        "clip_id": clip_id,
        "modality": modality,
        "status": status,
        "features": features or {},
        "predictions": predictions or {},
        "quality_metrics": quality_metrics or {},
        "timestamps": {},
    }


def test_all_arrive_early():
    gate = SyncGate(timeout=5)  # short timeout for test speed
    clip = "clip123"
    # Send macro, rppg, mer results quickly
    gate.receive(make_result(clip, "macro"))
    gate.receive(make_result(clip, "rppg"))
    bundle = gate.receive(make_result(clip, "mer"))  # returns bundle on early release
    assert bundle is not None
    assert bundle["availability"]["macro"]
    assert bundle["availability"]["rppg"]
    assert bundle["availability"]["mer"]


def test_missing_one_modality():
    gate = SyncGate(timeout=1)
    clip = "clip_missing"
    gate.receive(make_result(clip, "macro"))
    gate.receive(make_result(clip, "rppg"))
    # Wait for timeout to trigger release with mer missing
    time.sleep(1.2)
    bundle = gate.check_and_release(clip)
    assert bundle is not None
    assert bundle["availability"]["macro"]
    assert bundle["availability"]["rppg"]
    assert not bundle["availability"]["mer"]


def test_late_arrival_discard():
    gate = SyncGate(timeout=0.5)
    clip = "clip_late"
    gate.receive(make_result(clip, "macro"))
    # Let timeout fire
    time.sleep(0.6)
    bundle = gate.check_and_release(clip)
    assert bundle is not None
    # Now send a late result for rppg – should be logged as late but not affect bundle
    late = make_result(clip, "rppg")
    gate.receive(late)  # internal logging only
    # Ensure no second bundle is created
    assert gate.check_and_release(clip) is None


def test_duplicate_result_ignored():
    gate = SyncGate(timeout=2)
    clip = "clip_dup"
    res = make_result(clip, "macro")
    gate.receive(res)
    # Send duplicate same modality again
    gate.receive(res)
    # Should still only have one entry
    assert len(gate._pending[clip]["results"]) == 1

if __name__ == "__main__":
    # Simple manual run
    test_all_arrive_early()
    test_missing_one_modality()
    test_late_arrival_discard()
    test_duplicate_result_ignored()
    print("All gate tests passed")
