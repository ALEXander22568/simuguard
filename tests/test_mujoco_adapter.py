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


def test_com_velocity_of_spinning_body():
    xml = """<mujoco><option timestep="0.0005" gravity="0 0 0"/><worldbody><body name="obj" pos="0 0 1"><freejoint/>
             <geom type="box" size="0.02 0.02 0.02" pos="0 0 0.05" mass="0.1"/></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[:] = [0.3, 0.0, 0.0, 0.0, 12.0, 0.0]  # spinning about y with the centre of mass 5 cm above the origin
    mujoco.mj_forward(model, data)
    state = MujocoAdapter(model, data, roles=TaskRoles(targets={"obj"})).read_states(["body:obj"])["body:obj"]
    before = data.xipos[1].copy()
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    finite_difference = (data.xipos[1] - before) / model.opt.timestep
    np.testing.assert_allclose(state.linear_velocity, finite_difference, atol=0.01)  # 0.9 m/s, not 1.5
    np.testing.assert_allclose(state.angular_velocity, [0.0, 12.0, 0.0])


def test_exact_replay_when_forward_runs_before_step():
    """robosuite 1.4 (LIBERO): sim.forward(); controller (another forward, then ctrl); sim.step()."""
    adapter = make()
    records, reference = [], []
    hook = adapter.install_substep_hook(
        lambda: reference.append(adapter.data.qpos.copy()),
        before=lambda: records.append(adapter.capture_control()),
    )
    start = adapter.capture_snapshot(0)
    rng = np.random.default_rng(1)
    for _ in range(300):
        mujoco.mj_forward(adapter.model, adapter.data)  # env loop, stale ctrl
        mujoco.mj_forward(adapter.model, adapter.data)  # OSC controller update
        adapter.data.ctrl[:] = rng.normal(0, 2.0, 1)
        adapter.step()
    hook.remove()
    assert sum("qacc_warmstart" in r.payload for r in records) > 250
    adapter.restore_snapshot(start, method="native")
    for record, expected in zip(records, reference):
        adapter.apply_control(record)
        adapter.step_physics()
        np.testing.assert_array_equal(adapter.data.qpos, expected)


def test_libero_roles_from_goal():
    from types import SimpleNamespace

    from simuguard.adapters.mujoco.libero import goal_predicates, libero_roles

    xml = """<mujoco><worldbody><geom type="plane" size="1 1 0.1"/>
      <body name="robot0_link0"><joint type="hinge" axis="0 0 1"/><geom type="box" size="0.02 0.02 0.02"/></body>
      <body name="fixed_mount0_base" pos="0 -0.5 0"><geom type="box" size="0.05 0.05 0.05"/></body>
      <body name="bowl_1_main" pos="0.1 0 0.1"><freejoint/><geom type="box" size="0.02 0.02 0.02"/></body>
      <body name="plate_1_main" pos="0.2 0 0.1"><freejoint/><geom type="box" size="0.03 0.03 0.005"/></body>
      <body name="cookies_1_main" pos="0.3 0 0.1"><freejoint/><geom type="box" size="0.02 0.02 0.02"/></body>
      <body name="cabinet_1_main" pos="0 0.4 0"><geom type="box" size="0.1 0.1 0.1"/>
        <body name="cabinet_1_drawer"><joint type="slide" axis="1 0 0"/><geom type="box" size="0.05 0.05 0.02"/></body></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    def env(goal):
        rs = SimpleNamespace(
            sim=SimpleNamespace(model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data)),
            objects_dict={n: SimpleNamespace(root_body=f"{n}_main") for n in ("bowl_1", "plate_1", "cookies_1")},
            fixtures_dict={"cabinet_1": SimpleNamespace(root_body="cabinet_1_main")},
            object_sites_dict={"cabinet_1_top_region": SimpleNamespace(parent_name="cabinet_1"),
                               "main_table_front_region": SimpleNamespace(parent_name=None)},
            parsed_problem={"goal_state": goal}, obj_of_interest=[])
        return SimpleNamespace(env=rs)

    roles, info = libero_roles(env([["on", "bowl_1", "plate_1"]]))
    assert roles.targets == {"bowl_1_main"} and roles.containers == {"plate_1_main"}
    adapter = MujocoAdapter(model, data, roles=roles)
    bodies = adapter.bodies()
    assert bodies["body:cookies_1_main"].role == BodyRole.OBJECT
    assert bodies["body:fixed_mount0_base"].role == BodyRole.ROBOT
    roles, _ = libero_roles(env([["in", "bowl_1", "cabinet_1_top_region"], ["close", "cabinet_1"]]))
    assert roles.containers == {"cabinet_1_main", "cabinet_1_drawer"}
    roles, _ = libero_roles(env([["on", "plate_1", "main_table_front_region"]]))
    assert roles.targets == {"plate_1_main"} and roles.containers == set()  # the table stays scenery
    assert goal_predicates([["And", ["On", "a", "b"], ["Open", "c"]]]) == [["on", "a", "b"], ["open", "c"]]


def test_model_fingerprint_and_placement_restore():
    import json

    from simuguard.adapters.mujoco.libero import model_fingerprint, restore_placement

    recorded = mujoco.MjModel.from_xml_string(XML)
    rng = np.random.default_rng(3)
    recorded.body_pos[1] += rng.normal(0, 0.01, 3)  # a fixture placed by a random sampler on reset
    fingerprint = json.loads(json.dumps(model_fingerprint(recorded)))  # as stored in the manifest
    rebuilt = mujoco.MjModel.from_xml_string(XML)
    assert model_fingerprint(rebuilt)["sha256"] != fingerprint["sha256"]
    assert restore_placement(rebuilt, fingerprint) > 0
    assert model_fingerprint(rebuilt)["sha256"] == fingerprint["sha256"]


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
