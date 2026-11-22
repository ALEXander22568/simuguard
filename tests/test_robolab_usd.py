"""RoboLab adapter pieces that need only USD (pxr), not Isaac Sim."""

import unittest

try:
    from pxr import Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None

from simuguard.adapters.isaac.robolab import body_prim


@unittest.skipIf(Usd is None, "needs pxr (usd-core)")
class BodyPrimTest(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        for path in ("/World/envs/env_3/scene/banana", "/World/envs/env_3/robot/Gripper/Robotiq_2F_85/left_inner_finger"):
            UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(self.stage, path).GetPrim())
        UsdGeom.Xform.Define(self.stage, "/World/envs/env_3/robot/Gripper/Robotiq_2F_85/Joints/left_inner_finger")

    def test_scene_object(self):
        self.assertEqual(str(body_prim(self.stage, 3, "banana").GetPath()), "/World/envs/env_3/scene/banana")

    def test_robot_link_is_the_nested_rigid_body(self):
        prim = body_prim(self.stage, 3, "robot/left_inner_finger")
        self.assertEqual(str(prim.GetPath()), "/World/envs/env_3/robot/Gripper/Robotiq_2F_85/left_inner_finger")

    def test_missing_bodies_raise(self):
        for body_id in ("apple", "robot/right_inner_finger"):
            with self.assertRaises(ValueError):
                body_prim(self.stage, 3, body_id)
        with self.assertRaises(ValueError):
            body_prim(self.stage, 0, "banana")


if __name__ == "__main__":
    unittest.main()
