"""Isaac Sim adapter pieces that do not need Isaac Sim (tensor layouts, contact merging)."""

import importlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

from simuguard.adapters.isaac.physx_io import (
    PathResolver,
    changed_entries,
    merge_contact_records,
    pose_wxyz_to_xyzw,
    pose_xyzw_to_wxyz,
    states_from_arrays,
)
from simuguard.core.controls import ControlLog
from simuguard.core.snapshot_types import ControlRecord


def point(pos, normal, impulse, sep=0.0):
    return SimpleNamespace(
        position=SimpleNamespace(x=pos[0], y=pos[1], z=pos[2]),
        normal=SimpleNamespace(x=normal[0], y=normal[1], z=normal[2]),
        impulse=np.asarray(impulse, float),  # both carb.Float3-like and array-like inputs are accepted
        separation=sep,
    )


def header(a0, a1, offset, count):
    return SimpleNamespace(actor0=a0, actor1=a1, contact_data_offset=offset, num_contact_data=count)


class IsaacPureTests(unittest.TestCase):
    def test_import_does_not_require_isaac(self):
        importlib.import_module("simuguard.adapters.isaac")
        for name in ("omni", "pxr", "isaacsim", "isaaclab"):
            self.assertNotIn(name, sys.modules)

    def test_pose_layout_roundtrip(self):
        xyzw = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]])
        wxyz = pose_xyzw_to_wxyz(xyzw)
        np.testing.assert_array_equal(wxyz, [[1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3]])
        np.testing.assert_array_equal(pose_wxyz_to_xyzw(wxyz), xyzw)

    def test_states_from_arrays_selects_rows(self):
        poses = np.array([[0, 0, 0, 0, 0, 0, 1], [1, 2, 3, 0, 0, 0, 1.0]])
        vels = np.array([[0.0] * 6, [0.1, 0.2, 0.3, 1, 2, 3]])
        states = states_from_arrays(["b"], poses, vels, rows=[1])
        np.testing.assert_array_equal(states["b"].position, [1, 2, 3])
        np.testing.assert_array_equal(states["b"].quaternion, [1, 0, 0, 0])
        np.testing.assert_array_equal(states["b"].linear_velocity, [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(states["b"].angular_velocity, [1, 2, 3])

    def test_path_resolver_exact_prefix_and_fallback(self):
        resolver = PathResolver(
            exact={"/World/envs/env_0/rigid/bottle/b_0": "obj:bottle0"},
            prefixes={"/World/envs/env_0/Table": "static:table", "/World/envs/env_0/geometry/dustbin/d_0": "static:dustbin"},
        )
        self.assertEqual(resolver.resolve_path("/World/envs/env_0/rigid/bottle/b_0"), "obj:bottle0")
        self.assertEqual(resolver.resolve_path("/World/envs/env_0/Table/tbl/mesh"), "static:table")
        self.assertEqual(resolver.resolve_path("/World/envs/env_0/geometry/dustbin/d_0/collider_3"), "static:dustbin")
        # a collider below a rigid body resolves to the body
        self.assertEqual(resolver.resolve_path("/World/envs/env_0/rigid/bottle/b_0/collisions/mesh_0"), "obj:bottle0")
        self.assertTrue(resolver.resolve_path("/World/other/thing").startswith("unknown:"))
        self.assertIn("/World/other/thing", resolver.unknown)
        calls = []
        decode = lambda key: calls.append(key) or "/World/envs/env_0/Table"  # noqa: E731
        self.assertEqual(resolver.resolve(7, decode), "static:table")
        self.assertEqual(resolver.resolve(7, decode), "static:table")
        self.assertEqual(calls, [7])  # cached per encoded key

    def test_merge_contact_records_orients_impulse_on_body_a(self):
        names = {1: "obj:bottle0", 2: "static:table", 3: "link:robot0/link7"}
        points = [
            point([0, 0, 0.76], [0, 0, 1], [0, 0, 0.002]),  # bottle(actor0) on table: +z impulse on the bottle
            point([0.01, 0, 0.76], [0, 0, 1], [0, 0, 0.001]),
            point([0, 0.02, 0.8], [1, 0, 0], [0.003, 0, 0], -0.001),  # finger(actor0) pushes bottle(actor1)
        ]
        headers = [header(1, 2, 0, 2), header(3, 1, 2, 1), header(2, 3, 3, 0)]  # last: contact lost, no points
        pairs = {(p.body_a, p.body_b): p for p in merge_contact_records(headers, points, names.__getitem__)}
        self.assertEqual(set(pairs), {("obj:bottle0", "static:table"), ("link:robot0/link7", "obj:bottle0")})
        table = pairs[("obj:bottle0", "static:table")]
        np.testing.assert_allclose(table.total_impulse, [0, 0, 0.003])
        self.assertEqual(table.point_count, 2)
        self.assertAlmostEqual(table.force_estimate(0.004), 0.75)
        finger = pairs[("link:robot0/link7", "obj:bottle0")]
        np.testing.assert_allclose(finger.total_impulse, [0.003, 0, 0])  # link is body_a and actor0
        self.assertAlmostEqual(finger.max_penetration, 0.001)
        # flip: when the actor order disagrees with the sorted pair, the sign flips
        flipped = merge_contact_records([header(2, 1, 0, 1)], [points[0]], names.__getitem__)
        np.testing.assert_allclose(flipped[0].total_impulse, [0, 0, -0.002])

    def test_merge_contact_records_keep_and_self_pairs(self):
        names = {1: "a", 2: "b", 5: "a"}
        pts = [point([0, 0, 0], [0, 0, 1], [0, 0, 1])] * 3
        out = merge_contact_records([header(1, 5, 0, 1), header(1, 2, 1, 2)], pts, names.__getitem__, keep=lambda a, b: True)
        self.assertEqual(len(out), 1)  # a-a (two colliders of one body) dropped
        self.assertEqual(out[0].shape_pairs, 1)
        self.assertEqual(merge_contact_records([header(1, 2, 0, 1)], pts, names.__getitem__, keep=lambda a, b: False), [])

    def test_changed_entries(self):
        a = np.array([1.0, np.nan, 3.0])
        self.assertEqual(changed_entries(a, a.copy()).tolist(), [])
        b = a.copy()
        b[2] = 4.0
        self.assertEqual(changed_entries(b, a).tolist(), [2])
        self.assertEqual(changed_entries(np.zeros(2), np.zeros(3)).tolist(), [0, 1])

    def test_control_payload_roundtrips_exactly_through_control_log(self):
        payload = {
            "robot0": {"pos_target": np.float32([0.1, -0.2, 0.3]).astype(np.float64), "vel_target": np.zeros(3), "effort": np.zeros(3)},
            "__writes__": {"free_pose": {"idx": np.array([2, 9], dtype=np.int64), "val": np.array([0.5, -1.25])}},
        }
        log = ControlLog.from_records([ControlRecord(1, 0, payload), ControlRecord(2, 0, {"robot0": payload["robot0"]})])
        back = ControlLog.from_bytes(log.to_bytes())
        first = back.get(1).payload
        np.testing.assert_array_equal(np.float32(first["robot0"]["pos_target"]), np.float32([0.1, -0.2, 0.3]))
        self.assertEqual(first["__writes__"]["free_pose"]["idx"], [2, 9])
        self.assertEqual(back.get(2).payload.keys(), {"robot0"})


if __name__ == "__main__":
    unittest.main()
