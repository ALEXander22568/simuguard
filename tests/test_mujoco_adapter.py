"""MuJoCo adapter: contacts, roles and bit-exact replay (including state written between steps)."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from simuguard.adapters.mujoco import MujocoAdapter, TaskRoles  # noqa: E402
from simuguard.core.types import BodyKind, BodyRole  # noqa: E402

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="door" pos="0.3 0 0.2"><joint name="hinge" type="hinge" axis="0 0 1" damping="0.1"/>
      <geom type="box" size="0.1 0.01 0.1" mass="0.5"/></body>
    <body name="robot0_finger" pos="-0.3 0 0.3"><joint name="slide" type="slide" axis="0 0 1"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.2"/></body>
    <body name="cube" pos="0 0 0.05"><freejoint/><geom type="box" size="0.03 0.03 0.03" mass="0.05"/></body>
  </worldbody>
  <actuator><motor joint="slide" gear="1"/></actuator>
</mujoco>
"""


def make() -> MujocoAdapter:
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return MujocoAdapter(model, data, roles=TaskRoles(targets={"cube"}))


def test_roles_and_contacts():
    adapter = make()
    bodies = adapter.bodies()
    assert bodies["body:cube"].kind == BodyKind.DYNAMIC and bodies["body:cube"].role == BodyRole.TARGET
    assert bodies["body:robot0_finger"].role == BodyRole.ROBOT
    assert bodies["body:door"].kind == BodyKind.LINK
    for _ in range(200):
        adapter.step()
    pairs = adapter.read_contacts()
    assert any({p.body_a, p.body_b} == {"body:world", "body:cube"} for p in pairs)
    # resting cube: the normal impulse carries its weight
    pair = next(p for p in pairs if {p.body_a, p.body_b} == {"body:world", "body:cube"})
    total = sum(np.asarray(pt.impulse) for pt in pair.points)
    assert abs(abs(total[2]) / adapter.timestep() - 0.05 * 9.81) < 0.05


def test_exact_replay_with_state_written_between_steps():
    adapter = make()
    records, reference = [], []
    hook = adapter.install_substep_hook(
        lambda: reference.append(adapter.data.qpos.copy()),
        before=lambda: records.append(adapter.capture_control()),
    )
    start = adapter.capture_snapshot(0)
    rng = np.random.default_rng(0)
    for step in range(300):
        adapter.data.ctrl[:] = rng.normal(0, 2.0, 1)
        if step % 25 == 0:  # benchmark code outside the physics step moves a fixture joint
            adapter.data.qpos[adapter.model.jnt_qposadr[0]] += 0.01
        adapter.step()
    hook.remove()
    assert any("qpos_idx" in r.payload for r in records)
    adapter.restore_snapshot(start, method="native")
    for record, expected in zip(records, reference):
        adapter.apply_control(record)
        adapter.step_physics()
        np.testing.assert_array_equal(adapter.data.qpos, expected)
