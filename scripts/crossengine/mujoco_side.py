#!/usr/bin/env python3
"""MuJoCo side of the cross-engine contact experiment.

Rebuilds the scenes of sapien_side.py from the exported geometry (identical convex pieces, masses,
inertias and COM frames), runs the same scenarios under MuJoCo settings variants and writes the
same per-step quantities.  MuJoCo takes the convex hull of every mesh geom, as PhysX does for every
piece added with add_multiple_convex_collisions_from_file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def quat_to_mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


def fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in v)


def mesh_assets(prefix: str, pieces: list, min_extent: float = 1e-5) -> tuple[list[str], list[str], int]:
    assets, geoms, dropped = [], [], 0
    for i, piece in enumerate(pieces):
        verts = np.asarray(piece, dtype=np.float64)
        ext = verts.max(axis=0) - verts.min(axis=0)
        if ext.min() < min_extent:  # a flat piece has no volume; MuJoCo cannot build its hull
            dropped += 1
            continue
        assets.append(f'<mesh name="{prefix}{i}" vertex="{fmt(verts.reshape(-1))}"/>')
        geoms.append(f'<geom type="mesh" mesh="{prefix}{i}"/>')
    return assets, geoms, dropped


def body_xml(name: str, geom: dict, pos, quat, geoms: list[str], gravcomp: bool = False) -> str:
    gc = ' gravcomp="1"' if gravcomp else ""
    return (f'<body name="{name}" pos="{fmt(pos)}" quat="{fmt(quat)}"{gc}><freejoint/>'
            f'<inertial pos="{fmt(geom["com_pos"])}" quat="{fmt(geom["com_quat_wxyz"])}" mass="{geom["mass"]:.9g}" '
            f'diaginertia="{fmt(geom["inertia"])}"/>' + "".join(geoms) + "</body>")


def build_model(scenario: dict, geometry: dict, cfg: dict) -> tuple[mujoco.MjModel, dict]:
    ts = cfg.get("timestep", 0.004)
    solref = cfg.get("solref")
    default_geom = 'friction="0.5 0.005 0.0001" condim="3"' + (f' solref="{fmt(solref)}"' if solref else "")
    flags = '<flag refsafe="disable"/>' if cfg.get("refsafe", True) is False else ""
    info: dict = {}
    if scenario["kind"] == "canonical":
        t, h, m = scenario["wall_thickness"], scenario["cube_half"], scenario["mass"]
        inertia = [m * (2 * h) ** 2 / 6] * 3
        cube = {"com_pos": [0, 0, 0], "com_quat_wxyz": [1, 0, 0, 0], "mass": m, "inertia": inertia}
        body = body_xml("moving", cube, [t / 2 + h - scenario["overlap"], 0, 0.5], [1, 0, 0, 0],
                        [f'<geom type="box" size="{h} {h} {h}"/>'], gravcomp=not scenario.get("gravity", True))
        world = f'<geom name="wall" type="box" size="{t / 2} 0.2 0.2" pos="0 0 0.5"/>' + body
        assets = []
        info["other"] = "world"
    else:
        a_assets, a_geoms, a_drop = mesh_assets("basket", geometry["basket"]["pieces"])
        c_assets, c_geoms, c_drop = mesh_assets("can", geometry["can"]["pieces"])
        info["dropped_flat_pieces"] = {"basket": a_drop, "can": c_drop}
        bp, bq = scenario["basket"]["p"], scenario["basket"]["q"]
        rot = quat_to_mat(bq)
        z_top = min(float((np.asarray(p) @ rot.T + np.asarray(bp))[:, 2].min()) for p in geometry["basket"]["pieces"])
        table = f'<geom name="table" type="box" size="0.6 0.6 0.025" pos="{bp[0]} {bp[1]} {z_top - 0.025}"/>'
        world = (table + body_xml("container", geometry["basket"], bp, bq, a_geoms)
                 + body_xml("moving", geometry["can"], scenario["can"]["p"], scenario["can"]["q"], c_geoms))
        assets = a_assets + c_assets
        info["other"] = "container"
    xml = (f'<mujoco><compiler inertiafromgeom="false"/><option timestep="{ts}" gravity="0 0 -9.81">{flags}</option>'
           f'<default><geom {default_geom}/></default><asset>{"".join(assets)}</asset>'
           f'<worldbody>{world}</worldbody></mujoco>')
    return mujoco.MjModel.from_xml_string(xml), info


def set_velocity(model, data, body: str, state: dict) -> None:
    """SAPIEN gives the COM velocity and a world-frame angular velocity; a MuJoCo free joint wants the
    velocity of the body origin (world frame) and the angular velocity in the body frame."""
    b = model.body(body).id
    jnt = model.body_jntadr[b]
    adr = model.jnt_dofadr[jnt]
    rot = data.xmat[b].reshape(3, 3)
    w_world = np.asarray(state.get("w", [0, 0, 0]), dtype=np.float64)
    v_com = np.asarray(state.get("v", [0, 0, 0]), dtype=np.float64)
    com_world = data.xipos[b]
    v_origin = v_com - np.cross(w_world, com_world - data.xpos[b])
    data.qvel[adr:adr + 3] = v_origin
    data.qvel[adr + 3:adr + 6] = rot.T @ w_world


def com_speed(model, data, b: int, buf: np.ndarray) -> float:
    # mjOBJ_BODY is the body's inertial frame: the linear part is already the centre-of-mass velocity
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, buf, 0)
    return float(np.linalg.norm(buf[3:]))


def pair_penetration(model, data, b_moving: int, b_other: int) -> float:
    worst = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        bodies = {int(model.geom_bodyid[c.geom[0]]), int(model.geom_bodyid[c.geom[1]])}
        if bodies == {b_moving, b_other}:
            worst = max(worst, -float(c.dist))
    return worst


def run_variant(scenario: dict, variant: str, cfg: dict, geometry: dict) -> dict:
    model, info = build_model(scenario, geometry, cfg)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    b_moving = model.body("moving").id
    b_other = 0 if info["other"] == "world" else model.body("container").id
    check = {}
    if scenario["kind"] != "canonical":
        # geometry sanity: world AABB of the can's geoms must match the exported pieces at the same pose
        world = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != b_moving or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            m = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[m]: model.mesh_vertadr[m] + model.mesh_vertnum[m]]
            world.append(v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
        world = np.concatenate(world)
        rot = quat_to_mat(scenario["can"]["q"])
        verts = np.concatenate([np.asarray(p) for p in geometry["can"]["pieces"]]) @ rot.T + np.asarray(scenario["can"]["p"])
        check["can_aabb_max_abs_diff_m"] = float(max(np.abs(verts.min(axis=0) - world.min(axis=0)).max(),
                                                     np.abs(verts.max(axis=0) - world.max(axis=0)).max()))
        set_velocity(model, data, "moving", scenario["can"])
        set_velocity(model, data, "container", scenario["basket"])
        mujoco.mj_forward(model, data)
    buf = np.zeros(6)
    p0 = data.xipos[b_moving].copy()
    speeds, pens = [], []
    for _ in range(int(round(scenario.get("steps", 125) * 0.004 / model.opt.timestep))):
        mujoco.mj_step(model, data)
        speeds.append(com_speed(model, data, b_moving, buf))
        pens.append(pair_penetration(model, data, b_moving, b_other))
    return {
        "scenario": scenario["name"], "variant": variant, "engine": "mujoco", "mujoco": mujoco.__version__,
        "timestep": float(model.opt.timestep), "speed": speeds, "penetration": pens,
        "max_speed": max(speeds), "step_of_max": int(np.argmax(speeds)), "max_penetration": max(pens),
        "displacement": float(np.linalg.norm(data.xipos[b_moving] - p0)), "settings": cfg, "info": info, "check": check,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = json.loads(Path(args.scenarios).read_text())
    geometry = json.loads(Path(args.geometry).read_text())
    results = []
    for scenario in spec["scenarios"]:
        for variant, cfg in spec["mujoco_variants"].items():
            r = run_variant(scenario, variant, cfg, geometry)
            results.append(r)
            print(f"{scenario['name']:28s} {variant:24s} max speed {r['max_speed']:8.3f} m/s at step {r['step_of_max']:4d}  "
                  f"max pen {1000 * r['max_penetration']:7.2f} mm  displacement {100 * r['displacement']:6.2f} cm"
                  + (f"  [geometry check {r['check']}, dropped {r['info'].get('dropped_flat_pieces')}]" if r["check"] else ""), flush=True)
    Path(args.out).write_text(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
