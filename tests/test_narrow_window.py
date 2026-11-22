"""Narrow depenetration window: cap only inside the window, then each body's own value back.

Pure tests (fake adapter) run anywhere; the SAPIEN test runs where SAPIEN is installed and
checks the real adapter methods on a scene with non-default, per-body original values.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "robotwin" / "narrow"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from window import DepenWindow, WindowRule  # noqa: E402


class FakeAdapter:
    """Implements the adapter's per-body depenetration API on a dict; ``table`` is static."""

    def __init__(self, values):
        self.values = dict(values)
        self.static = {"table"}

    def max_depenetration_velocities(self, body_ids):
        return {b: (None if b in self.static else self.values[b]) for b in body_ids}

    def set_body_max_depenetration_velocity(self, body_ids, cap):
        originals = {}
        for b in body_ids:
            if b in self.static or b in originals:
                continue
            originals[b] = self.values[b]
            self.values[b] = float(cap)
        return originals

    def restore_max_depenetration_velocity(self, originals):
        for b, v in originals.items():
            self.values[b] = v
        return {b: self.values[b] for b in originals}


def drive(window, adapter, substeps, penetration=lambda s: 0.0):
    """Mimic the replay loop; returns the value of ``can`` while each substep's step runs."""

    seen = {}
    for s in substeps:
        window.before_step(adapter, s)
        seen[s] = adapter.values["can"]
        window.after_step(adapter, s, penetration(s))
    return seen


ORIGINALS = {"can": 1e32, "basket": 7.25, "finger": 3.5}


class FixedWindow(unittest.TestCase):
    def test_cap_only_inside_window_and_exact_restore(self):
        adapter = FakeAdapter(ORIGINALS)
        window = DepenWindow(WindowRule(t0=100, lead=25, mode="fixed", after=50), ["can", "basket", "table"], 1.0)
        seen = drive(window, adapter, range(1, 300))
        self.assertEqual(window.started_at, 75)
        self.assertEqual(window.ended_after, 150)
        self.assertTrue(all(seen[s] == 1e32 for s in range(1, 75)))
        self.assertTrue(all(seen[s] == 1.0 for s in range(75, 151)))
        self.assertTrue(all(seen[s] == 1e32 for s in range(151, 300)))
        self.assertEqual(adapter.values, ORIGINALS)  # every body back to its *own* value (not a default)
        report = window.report()
        self.assertTrue(report["restored_exactly"])
        self.assertEqual(report["capped_bodies"], ["can", "basket"])  # static body skipped
        self.assertEqual(report["values_during_window"], {"can": 1.0, "basket": 1.0})

    def test_other_bodies_untouched(self):
        adapter = FakeAdapter(ORIGINALS)
        window = DepenWindow(WindowRule(t0=100), ["can"], 0.5)
        drive(window, adapter, range(1, 120))
        self.assertEqual(adapter.values["basket"], 7.25)
        self.assertEqual(adapter.values["finger"], 3.5)

    def test_early_anchor_clamps_start_to_one(self):
        self.assertEqual(WindowRule(t0=10, lead=25).start, 1)


class ResolvedWindow(unittest.TestCase):
    def test_ends_after_n_quiet_substeps_from_t0(self):
        adapter = FakeAdapter(ORIGINALS)
        # deep penetration up to t0+3, quiet after
        window = DepenWindow(WindowRule(t0=100, lead=25, mode="resolved", n_resolved=5, threshold_m=0.001,
                                        max_after=250), ["can"], 1.0)
        seen = drive(window, adapter, range(1, 400), penetration=lambda s: 0.015 if 95 <= s <= 103 else 0.0)
        self.assertEqual(window.ended_after, 108)  # 104..108 quiet
        self.assertEqual(window.end_reason, "resolved")
        self.assertEqual(seen[108], 1.0)
        self.assertEqual(seen[109], 1e32)
        self.assertAlmostEqual(window.max_penetration_m, 0.015)
        self.assertEqual(adapter.values, ORIGINALS)

    def test_quiet_before_t0_does_not_count(self):
        adapter = FakeAdapter(ORIGINALS)
        window = DepenWindow(WindowRule(t0=100, mode="resolved", n_resolved=5), ["can"], 1.0)
        drive(window, adapter, range(1, 200))
        self.assertEqual(window.ended_after, 104)  # counted from t0 on: 100..104

    def test_capped_at_max_after(self):
        adapter = FakeAdapter(ORIGINALS)
        window = DepenWindow(WindowRule(t0=100, mode="resolved", n_resolved=5, max_after=250), ["can"], 1.0)
        drive(window, adapter, range(1, 600), penetration=lambda s: 0.01)
        self.assertEqual(window.ended_after, 350)
        self.assertEqual(window.end_reason, "max_after")
        self.assertEqual(adapter.values, ORIGINALS)


class PersistentWindow(unittest.TestCase):
    def test_never_restores(self):
        adapter = FakeAdapter(ORIGINALS)
        window = DepenWindow(WindowRule(t0=100, mode="persistent"), ["can"], 1.0)
        seen = drive(window, adapter, range(1, 300))
        self.assertEqual(seen[299], 1.0)
        self.assertIsNone(window.report()["restored_exactly"])


try:
    import sapien  # noqa: F401
    HAVE_SAPIEN = True
except Exception:  # noqa: BLE001
    HAVE_SAPIEN = False


@unittest.skipUnless(HAVE_SAPIEN, "SAPIEN not installed")
class RealAdapterMethods(unittest.TestCase):
    """The RoboTwin adapter's per-body methods on a real SAPIEN scene (no RoboTwin task needed)."""

    def test_per_body_set_and_restore(self):
        import sapien

        from simuguard.adapters.robotwin.adapter import RoboTwinAdapter

        scene = sapien.Scene()
        scene.set_timestep(0.004)
        scene.add_ground(0)
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
        a = builder.build(name="box_a")
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
        b = builder.build(name="box_b")
        comp_a = a.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        comp_b = b.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        comp_a.set_max_depenetration_velocity(4.5)
        comp_b.set_max_depenetration_velocity(2.25)

        class Env:
            pass

        adapter = RoboTwinAdapter.__new__(RoboTwinAdapter)
        handle = type("H", (), {})
        ha, hb = handle(), handle()
        ha.component, hb.component = comp_a, comp_b
        adapter._bodies = {"actor:box_a": ha, "actor:box_b": hb}
        originals = adapter.set_body_max_depenetration_velocity(["actor:box_a"], 1.0)
        self.assertEqual(originals, {"actor:box_a": 4.5})
        self.assertEqual(adapter.max_depenetration_velocities(["actor:box_a", "actor:box_b"]),
                         {"actor:box_a": 1.0, "actor:box_b": 2.25})
        for _ in range(3):
            scene.step()
        restored = adapter.restore_max_depenetration_velocity(originals)
        self.assertEqual(restored, {"actor:box_a": 4.5})
        self.assertEqual(comp_b.get_max_depenetration_velocity(), 2.25)
        self.assertEqual(adapter.max_depenetration_velocities(["missing"]), {"missing": None})


if __name__ == "__main__":
    unittest.main()
