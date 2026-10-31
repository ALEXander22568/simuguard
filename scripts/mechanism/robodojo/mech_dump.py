"""Fresh-process replay of a recorded RoboDojo segment with a raw dump around one event:
per contact point (collider paths, position, normal, separation, impulse) for the watched bodies, their
world transforms each substep, and once the collision geometry of the watched prims as authored in USD.
Wraps simuguard.integrations.robodojo_eval without changing it.
env: MECH_WINDOW=lo:hi (substeps), MECH_NAMES=comma separated path fragments, MECH_OUT=/runs/dump.json
"""
import json, os, sys, traceback
import numpy as np

LO, HI = (int(x) for x in os.environ.get("MECH_WINDOW", "3860:3903").split(":"))
NAMES = [n for n in os.environ.get("MECH_NAMES", "").split(",") if n]
OUT = os.environ.get("MECH_OUT", "/runs/dump.json")
D = {"window": [LO, HI], "names": NAMES, "steps": {}, "geometry": None, "info": {}, "errors": []}
state = {"substep": None}


class _Done(Exception):
    pass


def _save():
    with open(OUT, "w") as fh:
        json.dump(D, fh)


def _m(mat):
    return [[float(mat[i][j]) for j in range(4)] for i in range(4)]


def _roots(stage):
    roots = {}
    paths = [str(p.GetPath()) for p in stage.Traverse()]
    D["matching_paths"] = [p for p in paths if any(k in p.lower() for k in ("tube", "slot", "wrench", "toolbox", "plier", "hammer", "tape"))][:400]
    for n in NAMES:
        cand = [p for p in paths if n.lower() in p.lower()]
        if cand:
            roots[n] = min(cand, key=len)
    return roots


def _geometry(stage, roots):
    from pxr import Usd, UsdGeom, UsdPhysics
    out = {}
    for name, root in roots.items():
        items = []
        rp = stage.GetPrimAtPath(root)
        for prim in Usd.PrimRange(rp, Usd.TraverseInstanceProxies()):
            rec = {"path": str(prim.GetPath()), "type": str(prim.GetTypeName()), "schemas": [str(s) for s in prim.GetAppliedSchemas()],
                   "instance_proxy": bool(prim.IsInstanceProxy())}
            attrs = {}
            for a in prim.GetAttributes():
                nm = a.GetName()
                if nm.startswith(("physics:", "physx")) or nm in ("size", "radius", "height", "axis", "extent", "xformOpOrder"):
                    try:
                        v = a.Get()
                        attrs[nm] = None if v is None else (v if isinstance(v, (bool, int, float, str)) else str(v)[:300])
                    except Exception as exc:  # noqa: BLE001
                        attrs[nm] = f"<{exc}>"
            rec["attrs"] = attrs
            if prim.IsA(UsdGeom.Xformable):
                rec["world"] = _m(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                pts = mesh.GetPointsAttr().Get()
                rec["points"] = [[float(c) for c in p] for p in pts] if pts is not None else None
                cnt = mesh.GetFaceVertexCountsAttr().Get(); idx = mesh.GetFaceVertexIndicesAttr().Get()
                rec["face_counts"] = [int(c) for c in cnt] if cnt is not None else None
                rec["face_indices"] = [int(c) for c in idx] if idx is not None else None
            items.append(rec)
        out[name] = {"root": root, "prims": items}
    return out


def _info():
    info = {}
    try:
        import omni.kit.app
        mgr = omni.kit.app.get_app().get_extension_manager()
        for ext in mgr.get_extensions():
            if ext.get("enabled") and ext["id"].split("-")[0] in ("omni.physx", "omni.physx.tensors", "omni.usdphysics", "isaacsim.core.version", "omni.physx.cooking"):
                info[ext["id"]] = ext.get("path")
        info["kit_version"] = omni.kit.app.get_app().get_build_version()
    except Exception as exc:  # noqa: BLE001
        info["ext_error"] = repr(exc)
    try:
        import carb
        s = carb.settings.get_settings()
        for k in ("/physics/cudaDevice", "/persistent/physics/useFastCache", "/physics/collisionApproximateCones", "/physics/collisionApproximateCylinders",
                  "/persistent/physics/defaultContactOffset", "/physics/suppressReadback", "/physics/updateToUsd"):
            info[k] = s.get(k)
    except Exception as exc:  # noqa: BLE001
        info["settings_error"] = repr(exc)
    try:
        import subprocess, glob
        for key, path in list(info.items()):
            if key.startswith("omni.physx-") and path:
                libs = [os.path.join(dp, f) for dp, _, fs in os.walk(path) for f in fs if "physx" in f.lower() and f.endswith(".so")]
                info["ext_top"] = sorted(os.listdir(path))[:40]
                info["physx_libs"] = sorted(os.path.basename(x) for x in libs)[:20]
                hdr = glob.glob(path + "/**/PxPhysicsVersion.h", recursive=True)
                if hdr:
                    txt = open(hdr[0]).read()
                    import re
                    info["PxPhysicsVersion.h"] = re.findall(r"define PX_PHYSICS_VERSION_(?:MAJOR|MINOR|BUGFIX)\s+(\d+)", txt)
                for lib in libs[:6]:
                    try:
                        o = subprocess.run("strings -n 5 '%s' | grep -m 8 -E 'PhysX[ -]?(SDK)?[ -]?5\\.[0-9]|PhysX-5|physx-5\\.|5\\.[0-9]+\\.[0-9]+\\.[0-9a-f]{6,}|PhysXGpu|PX_PHYSICS_VERSION'" % lib, shell=True, capture_output=True, text=True, timeout=60).stdout
                        if o.strip():
                            info.setdefault("physx_version_strings", {})[os.path.basename(lib)] = o.strip().splitlines()
                    except Exception as exc:  # noqa: BLE001
                        info.setdefault("strings_errors", []).append(repr(exc))
    except Exception as exc:  # noqa: BLE001
        info["lib_error"] = repr(exc)
    try:
        import omni.physx
        ci = omni.physx.get_physx_cooking_interface()
        info["cooking_interface"] = [n for n in dir(ci) if not n.startswith("_")]
        info["physx_interface"] = [n for n in dir(omni.physx.get_physx_interface()) if not n.startswith("_")][:120]
    except Exception as exc:  # noqa: BLE001
        info["cooking_error"] = repr(exc)
    return info


def install():
    from simuguard.adapters.isaac import adapter as A
    cls = A.IsaacAdapter
    orig_apply, orig_step = cls.apply_control, cls.step_physics

    def apply_control(self, control):
        state["substep"] = int(control.substep)
        return orig_apply(self, control)

    def step_physics(self):
        orig_step(self)
        k = state["substep"]
        if k is None or k < LO:
            return
        try:
            import omni.usd
            from omni.physx import get_physx_simulation_interface
            from pxr import PhysicsSchemaTools, Usd, UsdGeom
            stage = omni.usd.get_context().get_stage()
            if D["geometry"] is None:
                roots = _roots(stage); D["roots"] = roots
                D["geometry"] = _geometry(stage, roots); D["info"] = _info(); _save()
            roots = D["roots"]
            headers, data = get_physx_simulation_interface().get_contact_report()
            rows = []
            for h in headers:
                a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0)); a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
                if not any(r in a0 or r in a1 for r in roots.values()):
                    continue
                c0 = str(PhysicsSchemaTools.intToSdfPath(h.collider0)); c1 = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
                pts = []
                for p in data[h.contact_data_offset: h.contact_data_offset + h.num_contact_data]:
                    pts.append({"p": [float(p.position.x), float(p.position.y), float(p.position.z)], "n": [float(p.normal.x), float(p.normal.y), float(p.normal.z)],
                                "imp": [float(p.impulse.x), float(p.impulse.y), float(p.impulse.z)], "sep": float(p.separation),
                                "f0": int(getattr(p, "face_index0", -1)), "f1": int(getattr(p, "face_index1", -1))})
                rows.append({"actor0": a0, "actor1": a1, "collider0": c0, "collider1": c1, "type": str(getattr(h, "type", "")), "points": pts})
            poses = {}
            for name, root in roots.items():
                prim = stage.GetPrimAtPath(root)
                poses[name] = {"root_world": _m(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))}
                sub = {}
                for q in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                    if q.IsA(UsdGeom.Mesh) or "PhysicsRigidBodyAPI" in [str(s) for s in q.GetAppliedSchemas()]:
                        sub[str(q.GetPath())] = _m(UsdGeom.Xformable(q).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                poses[name]["prims"] = sub
            try:
                st = self.read_states([b for b in self.bodies() if b.startswith("obj:")])
                bodies = {b: {"p": [float(x) for x in s.position], "q": [float(x) for x in s.quaternion], "v": [float(x) for x in s.linear_velocity], "w": [float(x) for x in s.angular_velocity]} for b, s in st.items()}
            except Exception as exc:  # noqa: BLE001
                bodies = {"error": repr(exc)}
            D["steps"][str(k)] = {"contacts": rows, "poses": poses, "bodies": bodies}
        except Exception:  # noqa: BLE001
            D["errors"].append({"substep": k, "tb": traceback.format_exc()[-1500:]})
        if k >= HI:
            _save()
            print(f"[mech] dump written to {OUT}: {len(D['steps'])} steps, errors {len(D['errors'])}", flush=True)
            raise _Done("window done")

    cls.apply_control = apply_control
    cls.step_physics = step_physics


if __name__ == "__main__":
    from simuguard.integrations import robodojo_eval as RE
    install()
    code = RE.main()
    sys.stdout.flush()
    os._exit(code)
