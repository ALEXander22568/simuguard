import tempfile
import unittest
from pathlib import Path

from simuguard.core.snapshot import (
    ControlLog,
    ControlRecord,
    ReplayBundle,
    Snapshot,
    SnapshotRing,
    build_replay_bundle,
    compare_states,
)


def snap(substep, native=b"\x00\x01"):
    return Snapshot(substep, substep // 10, {"a": [1.0, 2.0]}, ControlRecord(substep, 0, {"u": [substep]}), native, "raw")


class SnapshotTests(unittest.TestCase):
    def test_ring_eviction_and_pinning(self):
        ring = SnapshotRing(capacity=2, interval_substeps=10)
        for s in (0, 10, 20):
            ring.add(snap(s))
        self.assertEqual(ring.substeps(), [10, 20])
        ring.pin_before(15)
        ring.add(snap(30))
        ring.add(snap(40))
        self.assertEqual(ring.substeps(), [10, 30, 40])
        self.assertEqual(ring.latest_at_or_before(25).substep, 10)

    def test_control_log_requires_contiguous_coverage(self):
        log = ControlLog()
        for s in range(1, 6):
            log.append(ControlRecord(s, 0, {}))
        self.assertEqual([r.substep for r in log.between(2, 5)], [3, 4, 5])
        with self.assertRaises(LookupError):
            log.between(0, 7)
        with self.assertRaises(ValueError):
            log.append(ControlRecord(5, 0, {}))

    def test_serialization_roundtrip_verifies_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = snap(20, native=bytes(range(256))).save(Path(tmp) / "s.json.gz")
            loaded = Snapshot.load(path)
            self.assertEqual(loaded.native_state, bytes(range(256)))
            payload = loaded.to_dict()
            payload["public_state"]["a"][0] = 9.0
            with self.assertRaises(ValueError):
                Snapshot.from_dict(payload)

    def test_bundle_build_and_roundtrip(self):
        ring = SnapshotRing(capacity=4, interval_substeps=10)
        log = ControlLog()
        ring.add(snap(0))
        for s in range(1, 31):
            log.append(ControlRecord(s, 0, {"u": [s]}))
            if s % 10 == 0:
                ring.add(snap(s))
        bundle = build_replay_bundle(ring, log, trigger_substep=25, end_substep=30, trigger={"id": "x"})
        self.assertEqual(bundle.start_substep, 20)
        self.assertEqual([c.substep for c in bundle.controls], list(range(21, 31)))
        with tempfile.TemporaryDirectory() as tmp:
            again = ReplayBundle.load(bundle.save(Path(tmp) / "b.json.gz"))
        self.assertEqual(again.end_substep, 30)
        self.assertEqual(again.trigger, {"id": "x"})

    def test_compare_states(self):
        diff = compare_states({"x": [0.0, 1.0], "y": {"z": 2}}, {"x": [0.0, 1.5], "y": {"z": 2}, "w": 1})
        self.assertAlmostEqual(diff["max_abs_error"], 0.5)
        self.assertEqual(diff["structural_differences"][0]["path"], "w")


if __name__ == "__main__":
    unittest.main()
