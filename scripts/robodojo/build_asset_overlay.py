#!/usr/bin/env python3
"""Build a RoboDojo ``Assets`` overlay without touching the shared (read-only) asset tree.

The shared tree on 4090-node3 (``/data/shared/geshijia/urai-robodojo/assets``) holds only what the
URAI task lines needed: many object folders of the official benchmark are empty, and
``Robots/*/curobo.yml`` (generated from ``curobo_tmp.yml`` by RoboDojo's installer) is absent.
The overlay

* symlinks every top-level entry of the shared tree except ``Robots`` and ``Object``;
* copies ``Robots`` and writes ``curobo.yml`` from each ``curobo_tmp.yml`` with
  ``${ASSETS_PATH}`` -> the RoboDojo root inside the container (``/workspace/RoboDojo``);
* mirrors ``Object`` down to the category folders that the requested tasks use (every child is a
  symlink into the shared tree) and downloads the object folders that are missing there from the
  official ModelScope mirror of the RoboDojo dataset (``RoboDojo-Benchmark/RoboDojo``, ``Assets/``).

The container mounts the overlay at ``/workspace/RoboDojo/Assets`` and the shared tree at its own
host path, so the symlinks resolve inside the container.

Usage (host python, stdlib only)::

    python3 build_asset_overlay.py --tasks put_bottles_into_dustbin insert_tubes fill_pen_holder
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

MS_API = "https://modelscope.cn/api/v1/datasets/RoboDojo-Benchmark/RoboDojo"
LAYOUT_TYPES = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")


def ms_list(path: str) -> list[dict]:
    query = urllib.parse.urlencode({"Revision": "master", "Root": path, "Recursive": "false", "PageNumber": 1, "PageSize": 500})
    with urllib.request.urlopen(f"{MS_API}/repo/tree?{query}", timeout=60) as fh:
        data = json.load(fh)
    return (data.get("Data") or {}).get("Files") or []


def ms_download(path: str, target: Path, size: int | None = None) -> None:
    query = urllib.parse.urlencode({"Revision": "master", "FilePath": path})
    tmp = target.with_name(target.name + ".part")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{MS_API}/repo?{query}", timeout=300) as fh, tmp.open("wb") as out:
                shutil.copyfileobj(fh, out, 1 << 20)
            if size is not None and tmp.stat().st_size != size:
                raise IOError(f"size {tmp.stat().st_size} != {size}")
            tmp.replace(target)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"    retry {attempt + 1} for {path}: {exc}", flush=True)
            time.sleep(5)
    raise RuntimeError(f"download failed: {path}")


def needed_objects(base: Path, tasks: list[str], config: str, eval_seed: int) -> set[tuple[str, str, int]]:
    need: set[tuple[str, str, int]] = set()
    for task in tasks:
        files = glob.glob(str(base / "Eval_Layout" / "RoboDojo" / config / str(eval_seed) / f"{task}_*.json"))
        if not files:
            print(f"no layouts for {task}", file=sys.stderr)
        for f in files:
            data = json.loads(Path(f).read_text())
            for key in LAYOUT_TYPES:
                for cat, insts in (data.get(key) or {}).items():
                    for inst in insts:
                        sub = "Clutter" if inst.get("type") == "cluttered" else key
                        need.add((sub, cat, int(inst["category_idx"])))
    return need


def mirror_dir(real: Path, link_target: Path) -> None:
    """Turn ``real`` into a directory whose children link to ``link_target``'s children."""

    if real.is_symlink():
        real.unlink()
    real.mkdir(parents=True, exist_ok=True)
    if link_target.is_dir():
        for child in link_target.iterdir():
            dst = real / child.name
            if not dst.exists() and not dst.is_symlink():
                dst.symlink_to(child)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.realpath(os.environ.get("SG_ASSETS_BASE", "/data/shared/zhoujingjing/urai-robodojo/assets")))
    ap.add_argument("--overlay", default=os.path.join(os.environ.get("SG_ROOT", "/data/shared/zhoujingjing/simuguard-robodojo"), "assets_overlay"))
    ap.add_argument("--tasks", nargs="+", default=[])
    ap.add_argument("--config", default="arx_x5")
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--container-root", default="/workspace/RoboDojo")
    args = ap.parse_args()
    base, overlay = Path(args.base), Path(args.overlay)
    overlay.mkdir(parents=True, exist_ok=True)
    for entry in base.iterdir():
        if entry.name in ("Robots", "Object"):
            continue
        dst = overlay / entry.name
        if not dst.exists() and not dst.is_symlink():
            dst.symlink_to(entry)
    if not (overlay / "Robots").is_dir():
        shutil.copytree(base / "Robots", overlay / "Robots", symlinks=True)
    for tmp in (overlay / "Robots").rglob("*_tmp.yml"):
        text = tmp.read_text().replace("${ASSETS_PATH}", args.container_root).replace("$ASSETS_PATH", args.container_root)
        tmp.with_name(tmp.name.replace("_tmp.yml", ".yml")).write_text(text)
    # Object: mirror down to the categories we need, link the rest
    obj_base, obj_over = base / "Object" / "RoboDojo", overlay / "Object" / "RoboDojo"
    if (overlay / "Object").is_symlink():
        (overlay / "Object").unlink()
    mirror_dir(overlay / "Object", base / "Object")
    mirror_dir(obj_over, obj_base)
    need = needed_objects(base, args.tasks, args.config, args.eval_seed)
    downloaded, present = 0, 0
    for sub, cat, idx in sorted(need):
        mirror_dir(obj_over / sub, obj_base / sub)
        cat_dir = obj_over / sub / cat
        mirror_dir(cat_dir, obj_base / sub / cat)
        if not (cat_dir / "map.json").exists():
            remote = [f for f in ms_list(f"Assets/Object/RoboDojo/{sub}/{cat}") if f["Name"] == "map.json"]
            if remote:
                ms_download(remote[0]["Path"], cat_dir / "map.json", remote[0].get("Size"))
        obj_dir = cat_dir / f"{idx:05d}"
        if (obj_dir / "metadata.json").exists():
            present += 1
            continue
        if obj_dir.is_symlink():
            obj_dir.unlink()  # empty folder in the shared tree
        obj_dir.mkdir(parents=True, exist_ok=True)
        files = ms_list(f"Assets/Object/RoboDojo/{sub}/{cat}/{idx:05d}")
        if not files:
            print(f"  not on ModelScope: {sub}/{cat}/{idx:05d}", flush=True)
            continue
        for f in files:
            if f["Type"] == "tree":
                continue  # nested folders are not used by RoboDojo's loader
            ms_download(f["Path"], obj_dir / f["Name"], f.get("Size"))
        downloaded += 1
        print(f"  downloaded {sub}/{cat}/{idx:05d} ({len(files)} files)", flush=True)
    print(f"objects needed {len(need)}, already present {present}, downloaded {downloaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
