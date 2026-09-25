#!/usr/bin/env python3
"""SAPIEN / PhysX side of the cross-engine contact experiment.

Builds the can and basket with RoboTwin's own loader (create_actor, convex pieces, set_mass) and
RoboTwin's scene settings (1/250 s, friction 0.5/0.5, restitution 0, SAPIEN defaults otherwise), then

  export     writes every collision piece (actor-frame vertices), mass, inertia and COM frame to
             JSON so the MuJoCo side builds the identical geometry;
  simulate   runs the scenarios of scenarios.json under PhysX settings variants and writes the
             can's speed and the can-container penetration at every step.

Run with RoboTwin's client environment from anywhere; ROBOTWIN_ROOT points at the RoboTwin tree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/mnt/nvme0/twinguar/simuguard/RoboTwin")
sys.path.insert(0, ROBOTWIN_ROOT)
os.chdir(ROBOTWIN_ROOT)  # RoboTwin's loader resolves assets/objects relative to the cwd

import sapien  # noqa: E402
from envs.utils.create_actor import create_actor  # noqa: E402

RIGID = sapien.physx.PhysxRigidDynamicComponent


def make_scene(timestep: float) -> sapien.Scene:
    scene = sapien.Scene()
    scene.set_timestep(timestep)
    scene.default_physical_material = scene.create_physical_material(0.5, 0.5, 0.0)
    return scene


def build_object(scene: sapien.Scene, name: str, model_id: int, mass: float):
    actor = create_actor(scene, sapien.Pose(), name, convex=True, is_static=False, model_id=model_id)
    actor.set_mass(mass)
    return actor.actor if hasattr(actor, "actor") else actor.entity if hasattr(actor, "entity") else actor


def entity_of(actor):
    for attr in ("actor", "entity"):
        if hasattr(actor, attr):
            return getattr(actor, attr)
    return actor


def rigid(entity) -> "sapien.physx.PhysxRigidDynamicComponent":
    return entity.find_component_by_type(RIGID)


def export_geometry(entity) -> dict:
    comp = rigid(entity)
    pieces = []
    for shape in comp.collision_shapes:
        verts = np.asarray(shape.vertices, dtype=np.float64) * np.asarray(shape.scale, dtype=np.float64)
        pose = shape.local_pose.to_transformation_matrix()
        verts = verts @ pose[:3, :3].T + pose[:3, 3]
        pieces.append(verts.tolist())
    cm = comp.cmass_local_pose
    return {
        "pieces": pieces, "mass": float(comp.mass), "inertia": [float(x) for x in comp.inertia],
        "com_pos": [float(x) for x in cm.p], "com_quat_wxyz": [float(x) for x in cm.q],
    }


def lowest_z(geom: dict, p, q) -> float:
    pose = sapien.Pose(p, q).to_transformation_matrix()
    zs = [(np.asarray(piece) @ pose[:3, :3].T + pose[:3, 3])[:, 2].min() for piece in geom["pieces"]]
    return float(min(zs))


def container_penetration(scene: sapien.Scene, a, b) -> float:
    worst = 0.0
    for contact in scene.get_contacts():
        bodies = {contact.bodies[0].entity, contact.bodies[1].entity}
        if bodies == {a, b}:
            for point in contact.points:
                worst = max(worst, -float(point.separation))
    return worst


def run_variant(spec: dict, variant: str, cfg: dict, geometry: dict | None) -> dict:
    scene = make_scene(spec.get("timestep", 1 / 250))
    out: dict = {"scenario": spec["name"], "variant": variant, "engine": "sapien-physx", "sapien": sapien.__version__}
    if spec["kind"] == "canonical":
        # static thin wall (normal +x) and a free cube overlapping it by `overlap` metres
        t, half = spec["wall_thickness"], spec["cube_half"]
        wb = scene.create_actor_builder()
        wb.add_box_collision(half_size=[t / 2, 0.2, 0.2])
        wb.set_physx_body_type("static")
        wall = wb.build(name="wall")
        wall.set_pose(sapien.Pose([0, 0, 0.5]))
        cb = scene.create_actor_builder()
        cb.add_box_collision(half_size=[half] * 3)
        cube = cb.build(name="cube")
        rigid(cube).set_mass(spec["mass"])
        cube.set_pose(sapien.Pose([t / 2 + half - spec["overlap"], 0, 0.5]))
        moving, other = cube, wall
        if not spec.get("gravity", True):
            rigid(cube).set_disable_gravity(True)
    else:
        basket = entity_of(create_actor(scene, sapien.Pose(), "110_basket", convex=True, model_id=spec["basket_model_id"]))
        can = entity_of(create_actor(scene, sapien.Pose(), "071_can", convex=True, model_id=spec["can_model_id"]))
        rigid(basket).set_mass(spec["basket_mass"])
        rigid(can).set_mass(spec["can_mass"])
        bp, bq = spec["basket"]["p"], spec["basket"]["q"]
        z_top = lowest_z(geometry["basket"], bp, bq) if geometry else bp[2]
        tb = scene.create_actor_builder()
        tb.add_box_collision(half_size=[0.6, 0.6, 0.025])
        tb.set_physx_body_type("static")
        table = tb.build(name="table")
        table.set_pose(sapien.Pose([bp[0], bp[1], z_top - 0.025]))
        basket.set_pose(sapien.Pose(bp, bq))
        can.set_pose(sapien.Pose(spec["can"]["p"], spec["can"]["q"]))
        rigid(basket).set_linear_velocity(spec["basket"].get("v", [0, 0, 0]))
        rigid(basket).set_angular_velocity(spec["basket"].get("w", [0, 0, 0]))
        rigid(can).set_linear_velocity(spec["can"].get("v", [0, 0, 0]))
        rigid(can).set_angular_velocity(spec["can"].get("w", [0, 0, 0]))
        moving, other = can, basket
    cap = cfg.get("max_depenetration_velocity")
    if cap is not None:
        for ent in scene.entities:
            comp = ent.find_component_by_type(RIGID)
            if comp is not None:
                comp.set_max_depenetration_velocity(float(cap))
    if cfg.get("solver_position_iterations"):
        for ent in scene.entities:
            comp = ent.find_component_by_type(RIGID)
            if comp is not None:
                comp.set_solver_position_iterations(int(cfg["solver_position_iterations"]))
    scene.update_render = lambda: None  # no rendering needed
    speeds, pens, pos = [], [], []
    comp = rigid(moving)
    p0 = np.asarray(moving.get_pose().p, dtype=np.float64)
    for _ in range(int(spec.get("steps", 125))):
        scene.step()
        v = np.asarray(comp.linear_velocity, dtype=np.float64)
        speeds.append(float(np.linalg.norm(v)))
        pens.append(container_penetration(scene, moving, other))
        pos.append(np.asarray(moving.get_pose().p, dtype=np.float64).tolist())
    out.update({
        "speed": speeds, "penetration": pens,
        "max_speed": max(speeds), "step_of_max": int(np.argmax(speeds)),
        "max_penetration": max(pens), "displacement": float(np.linalg.norm(np.asarray(pos[-1]) - p0)),
        "settings": cfg,
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("export", "simulate"))
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--geometry", required=True, help="JSON written by export, read by simulate")
    ap.add_argument("--out")
    args = ap.parse_args()
    spec = json.loads(Path(args.scenarios).read_text())

    if args.mode == "export":
        scene = make_scene(1 / 250)
        basket = entity_of(create_actor(scene, sapien.Pose(), "110_basket", convex=True, model_id=spec["basket_model_id"]))
        can = entity_of(create_actor(scene, sapien.Pose(), "071_can", convex=True, model_id=spec["can_model_id"]))
        rigid(basket).set_mass(spec["basket_mass"])
        rigid(can).set_mass(spec["can_mass"])
        geometry = {"basket": export_geometry(basket), "can": export_geometry(can),
                    "sapien": sapien.__version__, "basket_model_id": spec["basket_model_id"],
                    "can_model_id": spec["can_model_id"]}
        Path(args.geometry).write_text(json.dumps(geometry))
        print(f"exported basket {len(geometry['basket']['pieces'])} pieces (m={geometry['basket']['mass']}), "
              f"can {len(geometry['can']['pieces'])} pieces (m={geometry['can']['mass']}, I={geometry['can']['inertia']})")
        return 0

    geometry = json.loads(Path(args.geometry).read_text())
    results = []
    for scenario in spec["scenarios"]:
        base = {k: spec[k] for k in ("basket_model_id", "can_model_id", "basket_mass", "can_mass") if k in spec}
        scenario = {**base, **scenario}
        for variant, cfg in spec["sapien_variants"].items():
            r = run_variant(scenario, variant, cfg, geometry)
            results.append(r)
            print(f"{scenario['name']:28s} {variant:12s} max speed {r['max_speed']:8.3f} m/s at step {r['step_of_max']:3d}  "
                  f"max pen {1000 * r['max_penetration']:7.2f} mm  displacement {100 * r['displacement']:6.2f} cm", flush=True)
    Path(args.out).write_text(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
