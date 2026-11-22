"""RoboLab adapter pieces that do not need Isaac Sim (write log, contacts, caps, provenance)."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from simuguard.core.types import ContactPair

from simuguard.adapters.isaac.robolab_io import (
    CommandBuffer,
    ContactLog,
    DepenetrationCap,
    WriteLog,
    body_id_from_relative,
    cap_transitions,
    compare_state_arrays,
    contact_pairs_from_data,
    effective_ejection_thresholds,
    ejection_config,
    freeze_text,
    parse_cap,
    server_info,
    sha256_text,
    segment_dir,
    split_env_path,
    tree_sha256,
)


class PathTests(unittest.TestCase):
    def test_split_env_path(self):
        self.assertEqual(split_env_path("/World/envs/env_3/scene/banana"), (3, "scene/banana"))
        self.assertEqual(split_env_path("/World/envs/env_12/robot/panda_link0"), (12, "robot/panda_link0"))
        self.assertIsNone(split_env_path("/World/ground"))

    def test_body_id(self):
        self.assertEqual(body_id_from_relative("scene/banana"), "banana")
        self.assertEqual(body_id_from_relative("robot/panda_hand"), "robot/panda_hand")

    def test_segment_dir(self):
        seg = {"dir": "/path/to/simuguard-robolab/runs/formal_pi05/T/T/run_1/env_9/seg_0"}
        self.assertEqual(segment_dir(Path("/mirror/formal_pi05/T/T"), seg), Path("/mirror/formal_pi05/T/T/run_1/env_9/seg_0"))
        self.assertEqual(segment_dir(Path("/path/to/simuguard-robolab/runs/formal_pi05/T/T"), seg), Path(seg["dir"]))


class DetectorConfigTests(unittest.TestCase):
    def test_thresholds_scale_with_timestep(self):
        cfg = ejection_config()
        self.assertEqual(cfg["ejection"]["delta_v_reference_timestep_s"], 0.004)
        eff = effective_ejection_thresholds(cfg["ejection"], 1.0 / 120.0)
        self.assertAlmostEqual(eff["scale"], (1.0 / 120.0) / 0.004)
        at_ref = effective_ejection_thresholds(cfg["ejection"], 0.004)
        self.assertAlmostEqual(eff["min_delta_v_mps"], at_ref["min_delta_v_mps"] * eff["scale"])

    def test_user_value_kept(self):
        cfg = ejection_config({"ejection": {"delta_v_reference_timestep_s": 0.01, "min_delta_v_mps": 0.5}})
        self.assertEqual(cfg["ejection"]["delta_v_reference_timestep_s"], 0.01)
        self.assertEqual(cfg["ejection"]["min_delta_v_mps"], 0.5)


class WriteLogTests(unittest.TestCase):
    def test_roundtrip_keeps_dtypes_shapes_and_order(self):
        log = WriteLog()
        full = np.arange(4 * 13, dtype=np.float32).reshape(4, 13)
        log.append(1, "articulation:robot", "set_dof_position_targets", np.array([1, 3], dtype=np.int32), full)
        log.append(1, "sim_view", "update_articulations_kinematic")
        log.append(2, "rigid:banana", "set_transforms", np.array([0], dtype=np.int32), np.ones((1, 7), dtype=np.float32))
        log.append(2, "articulation:robot", "set_coms", np.array([2], dtype=np.int64), np.zeros((1, 18, 7), dtype=np.float32))
        log.append(3, "rigid:bowl", "set_disable_gravities", np.array([0, 1], dtype=np.int32), np.array([[1], [0]], dtype=np.uint8))
        with tempfile.TemporaryDirectory() as tmp:
            path = WriteLog.save(log.writes, Path(tmp) / "w.npz", {"run": 0})
            writes, meta = WriteLog.load(path)
        self.assertEqual(meta, {"run": 0})
        self.assertEqual([(w.substep, w.target, w.method) for w in writes], [(w.substep, w.target, w.method) for w in log.writes])
        np.testing.assert_array_equal(writes[0].data, full[[1, 3]])
        self.assertEqual(writes[0].data.dtype, np.float32)
        self.assertEqual(writes[0].indices.dtype, np.int32)
        self.assertIsNone(writes[1].data)
        self.assertIsNone(writes[1].indices)
        self.assertEqual(writes[3].data.shape, (1, 18, 7))
        self.assertEqual(writes[3].indices.dtype, np.int64)
        np.testing.assert_array_equal(writes[4].data, np.array([[1], [0]], dtype=np.uint8))
        self.assertEqual(log.substeps(), (1, 3))
        self.assertEqual(sorted(log.by_substep()), [1, 2, 3])

    def test_copy_on_append(self):
        log = WriteLog()
        data = np.zeros((2, 3))
        log.append(1, "t", "set_x", np.array([0, 1]), data)
        data[:] = 5
        self.assertEqual(log.writes[0].data.sum(), 0)

    def test_command_buffer(self):
        log = WriteLog()
        log.append(4, "articulation:robot", "set_dof_position_targets", np.array([1], dtype=np.int32), np.array([[0.1, 0.2]]))
        buf = CommandBuffer(2)
        buf.update(log.writes[0])
        self.assertIsNone(buf.payload(0)["position_targets"])
        self.assertEqual(buf.payload(1)["position_targets"], [0.1, 0.2])


class ContactTests(unittest.TestCase):
    def test_pairs_per_env_with_sign_and_skip(self):
        # 2 envs, sensor "bowl", filters [banana, table]
        counts = np.array([[1, 0], [1, 2]])
        starts = np.array([[0, 1], [1, 2]])
        forces = np.array([10.0, 20.0, 1.0, 2.0])
        points = np.arange(12, dtype=float).reshape(4, 3)
        normals = np.tile([0.0, 0.0, 1.0], (4, 1))
        seps = np.array([-0.001, -0.002, 0.0, -0.003])
        out = contact_pairs_from_data("bowl", ["banana", "table"], forces, points, normals, seps, counts, starts, 0.01)
        self.assertEqual(sorted(out), [0, 1])
        pair = out[0][0]
        self.assertEqual((pair.body_a, pair.body_b), ("banana", "bowl"))  # sorted ids, sensor is body_b
        np.testing.assert_allclose(pair.points[0].impulse, [0, 0, -0.1])
        table = [p for p in out[1] if "table" in (p.body_a, p.body_b)][0]
        self.assertEqual((table.body_a, table.body_b), ("bowl", "table"))
        self.assertEqual(table.point_count, 2)
        skipped = contact_pairs_from_data("bowl", ["banana", "table"], forces, points, normals, seps, counts, starts, 0.01,
                                          skip_filters=[0])
        self.assertEqual([len(v) for _, v in sorted(skipped.items())], [1])

    def test_array_pair_matches_point_pair(self):
        out = contact_pairs_from_data("bowl", ["banana"], np.array([3.0, 4.0]), np.arange(6.0).reshape(2, 3),
                                      np.array([[0, 0, 1.0], [1.0, 0, 0]]), np.array([-0.002, 0.001]),
                                      np.array([[2]]), np.array([[0]]), 0.01)
        pair = out[0][0]
        plain = ContactPair(pair.body_a, pair.body_b, pair.points, shape_pairs=pair.shape_pairs)
        self.assertEqual(pair.point_count, 2)
        np.testing.assert_allclose(pair.total_impulse, plain.total_impulse)
        self.assertAlmostEqual(pair.impulse_abs_sum, plain.impulse_abs_sum)
        self.assertAlmostEqual(pair.max_penetration, plain.max_penetration)
        self.assertEqual(pair.to_dict(0.01), plain.to_dict(0.01))

    def test_contact_log_save(self):
        out = contact_pairs_from_data("a", ["b"], np.array([1.0]), np.zeros((1, 3)), np.array([[0, 0, 1.0]]),
                                      np.array([-0.01]), np.array([[1]]), np.array([[0]]), 0.01)
        log = ContactLog()
        log.append(7, out[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = log.save(Path(tmp) / "c.npz")
            with np.load(path) as data:
                rows = data["rows"]
        self.assertEqual(rows.shape, (1, 13))
        self.assertEqual(rows[0, 0], 7)
        self.assertAlmostEqual(rows[0, 12], -0.01)


class CapTests(unittest.TestCase):
    def test_parse(self):
        cap = parse_cap("1.0,bodies=banana+bowl,envs=0+3,from=1200,to=1500")
        self.assertEqual(cap, DepenetrationCap(1.0, ("banana", "bowl"), (0, 3), 1200, 1500))
        self.assertEqual(parse_cap("0.5"), DepenetrationCap(0.5))
        with self.assertRaises(ValueError):
            parse_cap("1.0,from=10,to=10")
        with self.assertRaises(ValueError):
            parse_cap("1.0,colour=red")

    def test_window(self):
        cap = DepenetrationCap(1.0, ("banana",), (2,), 10, 20)
        self.assertFalse(cap.active(9))
        self.assertTrue(cap.active(10))
        self.assertFalse(cap.active(20))
        self.assertTrue(cap.selects(2, "banana"))
        self.assertFalse(cap.selects(1, "banana"))
        self.assertFalse(cap.selects(2, "bowl"))
        self.assertEqual(cap_transitions([cap], 10), [(cap, True)])
        self.assertEqual(cap_transitions([cap], 20), [(cap, False)])
        self.assertEqual(cap_transitions([DepenetrationCap(1.0)], 5), [])


class ProvenanceTests(unittest.TestCase):
    def test_freeze_text_sorted_unique(self):
        text = freeze_text([("numpy", "1.26.0"), ("Torch", "2.5"), ("numpy", "1.26.0"), ("absl", "1")])
        self.assertEqual(text, "absl==1\nnumpy==1.26.0\nTorch==2.5\n")
        self.assertEqual(len(sha256_text(text)), 64)

    def test_server_info(self):
        self.assertIsNone(server_info(None))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.json"
            path.write_text(json.dumps({"pid": os.getpid(), "xla_python_client_mem_fraction": "0.45"}))
            self.assertEqual(server_info(path), {"pid": os.getpid(), "xla_python_client_mem_fraction": "0.45", "pid_alive": True})
            dead = subprocess.Popen([sys.executable, "-c", "pass"])
            dead.wait()
            path.write_text(json.dumps({"pid": dead.pid}))
            self.assertFalse(server_info(path)["pid_alive"])

    def test_tree_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "pkg" / "a.py").write_text("x = 1\n")
            (root / "b.py").write_text("y = 2\n")
            (root / "notes.txt").write_text("ignored")
            first = tree_sha256(root)
            (root / "notes.txt").write_text("changed")
            self.assertEqual(tree_sha256(root), first)
            (root / "pkg" / "a.py").write_text("x = 2\n")
            self.assertNotEqual(tree_sha256(root), first)

    def test_compare_state_arrays(self):
        a = np.array([[1.0, np.nan], [2.0, 3.0]])
        self.assertTrue(compare_state_arrays(a, a.copy())["bit_exact"])
        b = a.copy()
        b[1, 1] += 1e-7
        res = compare_state_arrays(a, b)
        self.assertFalse(res["bit_exact"])
        self.assertEqual(res["differing_values"], 1)
        self.assertAlmostEqual(res["max_abs_diff"], 1e-7)
        self.assertFalse(compare_state_arrays(a, a[:1])["bit_exact"])


if __name__ == "__main__":
    unittest.main()
