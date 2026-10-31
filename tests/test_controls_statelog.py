import unittest

import numpy as np

from simuguard.core.controls import ControlLog
from simuguard.core.snapshot import ControlRecord, ReplayBundle, Snapshot
from simuguard.core.statelog import StateLog
from simuguard.core.types import BodyState


def robotwin_like_payload(k: int, *, joints: int = 38, exotic: float | None = None) -> dict:
    rng = np.random.default_rng(k)
    targets = rng.standard_normal(joints).astype(np.float32).astype(float)
    payload = {
        "art:unnamed": {
            "qf": rng.standard_normal(joints).astype(np.float32).astype(float).tolist(),
            "joints": [
                {"index": i, "name": f"joint_{i}", "drive_target": [targets[i]], "drive_velocity_target": [0.0]}
                for i in range(joints)
            ],
        }
    }
    if exotic is not None:
        payload["art:unnamed"]["joints"][0]["drive_target"] = [exotic]
    return payload


class ControlLogTests(unittest.TestCase):
    def test_records_roundtrip_exactly_and_compactly(self):
        log = ControlLog()
        originals = [ControlRecord(s, s // 10, robotwin_like_payload(s)) for s in range(1, 301)]
        for record in originals:
            log.append(record)
        self.assertEqual(log.dtype, np.float32)
        self.assertEqual([r.payload for r in log.between(0, 300)], [r.payload for r in originals])
        self.assertEqual(log.get(150).payload, originals[149].payload)
        self.assertEqual(log.nbytes, 300 * 114 * 4)
        restored = ControlLog.from_bytes(log.to_bytes())
        self.assertEqual([r.payload for r in restored.records()], [r.payload for r in originals])
        self.assertEqual([r.control_step for r in restored.records()], [r.control_step for r in originals])

    def test_float64_promotion_keeps_exact_values(self):
        log = ControlLog()
        log.append(ControlRecord(1, 0, robotwin_like_payload(1)))
        exotic = 0.1  # not representable in float32
        log.append(ControlRecord(2, 0, robotwin_like_payload(2, exotic=exotic)))
        self.assertEqual(log.dtype, np.float64)
        self.assertEqual(log.get(2).payload["art:unnamed"]["joints"][0]["drive_target"], [exotic])
        self.assertEqual(log.get(1).payload, robotwin_like_payload(1))
        self.assertEqual(ControlLog.from_bytes(log.to_bytes()).get(2).payload, log.get(2).payload)

    def test_structure_change_uses_new_schema(self):
        log = ControlLog()
        log.append(ControlRecord(1, 0, robotwin_like_payload(1, joints=3)))
        log.append(ControlRecord(2, 0, robotwin_like_payload(2, joints=5)))
        log.append(ControlRecord(3, 0, {"kick": [0.0, 0.0, 3.0], "robot_v": [0, 0, 0], "mode": "x", "gain": 0.5}))
        restored = ControlLog.from_bytes(log.to_bytes())
        self.assertEqual(len(restored.get(2).payload["art:unnamed"]["joints"]), 5)
        self.assertEqual(restored.get(3).payload, {"kick": [0.0, 0.0, 3.0], "robot_v": [0, 0, 0], "mode": "x", "gain": 0.5})

    def test_gaps_and_order(self):
        log = ControlLog(maxlen=5)
        for s in range(1, 11):
            log.append(ControlRecord(s, 0, {"u": [float(s)]}))
        self.assertEqual(log.oldest_substep, 6)
        with self.assertRaises(LookupError):
            log.between(0, 10)
        with self.assertRaises(ValueError):
            log.append(ControlRecord(10, 0, {"u": [1.0]}))

    def test_bundle_embeds_compact_controls_and_reads_legacy(self):
        snap = Snapshot(0, 0, {"a": [1.0]}, ControlRecord(0, 0, {"u": [0.0]}))
        records = [ControlRecord(s, 0, robotwin_like_payload(s, joints=4)) for s in range(1, 21)]
        bundle = ReplayBundle(snap, records, [], {"id": "e"})
        data = bundle.to_dict()
        self.assertEqual(data["controls_encoding"], "control_log_npz_b64")
        self.assertEqual([r.payload for r in ReplayBundle.from_dict(data).controls], [r.payload for r in records])
        legacy = dict(data, controls=[r.to_dict() for r in records])
        legacy.pop("controls_encoding")
        self.assertEqual(len(ReplayBundle.from_dict(legacy).controls), 20)


class StateLogTests(unittest.TestCase):
    def test_roundtrip_full_precision(self):
        log = StateLog(["a", "b"])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        for s in range(1, 4):
            log.append(s, {"a": BodyState("a", np.array([0.1 * s, 1 / 3, 2.0]), q, np.ones(3), np.zeros(3))})
        restored = StateLog.from_bytes(log.to_bytes())
        np.testing.assert_array_equal(restored.positions("a"), log.positions("a"))
        self.assertTrue(np.isnan(restored.array()[0, 1, 0]))  # missing body stays NaN
        self.assertEqual(restored.substeps.tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
