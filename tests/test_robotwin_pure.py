"""RoboTwin adapter pieces that do not need SAPIEN."""

import importlib
import sys
import unittest

import numpy as np

from simuguard.adapters.robotwin.tasks import ContainmentGate, box_corners, get_task_spec, model_bounds, quat_wxyz_to_matrix
from simuguard.core.detectors.base import DetectorContext
from simuguard.core.types import BodyState, SubstepFrame


def state(body_id, p, q=(1, 0, 0, 0)):
    return BodyState(body_id, np.asarray(p, float), np.asarray(q, float), np.zeros(3), np.zeros(3))


class RoboTwinPureTests(unittest.TestCase):
    def test_import_does_not_require_sapien(self):
        module = importlib.import_module("simuguard.adapters.robotwin")
        self.assertTrue(hasattr(module, "RoboTwinAdapter"))
        if "sapien" not in sys.modules:
            self.assertNotIn("sapien", sys.modules)

    def test_task_spec_registry_and_overrides(self):
        spec = get_task_spec("place_can_basket")
        self.assertEqual(spec.target_attrs, ("can",))
        custom = get_task_spec("stack_bowls_two", {"target_attrs": ["bowl2"], "container_attrs": ["bowl1"]})
        self.assertEqual(custom.container_attrs, ("bowl1",))

    def test_quaternion_matrix_is_rotation(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        r = quat_wxyz_to_matrix(q)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(r @ np.array([1.0, 0, 0]), [0, 1.0, 0], atol=1e-12)

    def test_containment_gate(self):
        low, high = model_bounds({"center": [0, 0, 0.05], "extents": [0.2, 0.2, 0.1], "scale": [1, 1, 1]})
        t_low, t_high = model_bounds({"center": [0, 0, 0], "extents": [0.04, 0.04, 0.06], "scale": 1.0})
        gate = ContainmentGate("can", "basket", box_corners(t_low, t_high), low, high)
        ctx = DetectorContext("ep", 0.004, {})

        def at(p, q=(1, 0, 0, 0)):
            frame = SubstepFrame(1, 0.004, 0, 0.004, {"can": state("can", p), "basket": state("basket", (1, 1, 0), q)}, [])
            return gate(frame, "can", ctx)

        self.assertTrue(at((1, 1, 0.05))["active"])
        self.assertFalse(at((1, 1, 0.2))["active"])  # above the rim
        self.assertFalse(at((1.2, 1, 0.05))["active"])  # outside laterally
        self.assertFalse(gate(SubstepFrame(1, 0, 0, 0.004, {}, []), "basket", ctx)["applies"])


if __name__ == "__main__":
    unittest.main()
