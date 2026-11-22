#!/usr/bin/env python3
"""Environment fingerprint of the native SimuGuard x RoboDojo install (stdlib only).

usage: python3 fingerprint_robodojo.py [--policies P ...] [--full] [--out FILE]

Writes one JSON object.  Keys under "info" are expected to differ between machines
(host, kernel, GPU list); everything else should be identical on two machines that run
the same stack.  --full re-hashes every checkpoint file (else the coordinator's
SHA256SUMS is checked by size + mtime only and its own hash is reported).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("SGRD_ROOT", "/path/to/simuguard-robodojo"))
CKPT = Path(os.environ.get("CKPT_ROOT", "/path/to/robodojo-ckpt"))
RD = ROOT / "RoboDojo"
POL = RD / "XPolicyLab" / "policy"
CONDA_ENVS = {"InternW0_delta": "internw0-delta", "InternVLA_A1": "internvla_a1", "Xiaomi_Robotics_0": "mibot",
              "X_VLA": "XVLA", "OpenWAM": "openwam"}
UV_ENVS = {"GR00T_N17": POL / "GR00T_N17/gr00t_n17/.venv", "Pi_05": POL / "Pi_05/openpi/.venv",
           "Pi_0": POL / "Pi_0/openpi/.venv"}
TASKS = ("hang_mugs", "sweep_blocks", "classify_objects", "fill_pen_holder", "stack_bowls", "insert_tubes",
         "cover_blocks")


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERR {exc}"


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def git_state(path: Path) -> dict:
    out = {"head": sh(f"git -C {path} rev-parse HEAD"), "remote": sh(f"git -C {path} remote get-url origin")}
    diff = sh(f"git -C {path} diff HEAD")
    out["diff_sha256"] = sha_text(diff)
    out["dirty_files"] = len([x for x in sh(f"git -C {path} status --porcelain --untracked-files=no").splitlines() if x])
    subs = {}
    for line in sh(f"git -C {path} submodule status --recursive").splitlines():
        parts = line.strip().lstrip("+-U").split()
        if len(parts) >= 2:
            subs[parts[1]] = parts[0]
    out["submodules"] = subs
    return out


def tree_digest(folder: Path, pattern: str = "**/*") -> dict:
    files = sorted(p for p in folder.glob(pattern) if p.is_file())
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.relative_to(folder)}\0{sha_file(p)}\n".encode())
    return {"files": len(files), "sha256": h.hexdigest()}


def listing_digest(folder: Path) -> dict:
    """Path + size of every file (cheap; content hashing 39 GB of assets is not)."""
    h = hashlib.sha256()
    n = total = 0
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            h.update(f"{p.relative_to(folder)}\0{size}\n".encode())
            n += 1
            total += size
    return {"files": n, "bytes": total, "path_size_sha256": h.hexdigest()}


def freeze(policy: str) -> dict:
    if policy in UV_ENVS:
        text = sh(f"{ROOT}/bin/uv pip freeze --python {UV_ENVS[policy]}/bin/python")
    else:
        env = "RoboDojo" if policy == "RoboDojo" else CONDA_ENVS[policy]
        text = sh(f"{ROOT}/miniconda3/envs/{env}/bin/python -m pip freeze --all")
    lines = sorted(x for x in text.splitlines() if x)
    return {"packages": len(lines), "sha256": sha_text("\n".join(lines))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="*", default=list(UV_ENVS) + list(CONDA_ENVS))
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fp: dict = {"info": {
        "host": sh("hostname"), "kernel": sh("uname -r"),
        "gpus": sh("nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv,noheader"),
        "dkms": sh("dkms status 2>/dev/null"),
    }}
    fp["sys"] = {
        "os": sh(". /etc/os-release; echo $PRETTY_NAME"),
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u"),
        "cuda_driver_api": sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -1"),
        "vulkan_icd": sh("ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null | grep -i nvidia | sort -u | paste -sd,"),
        "uv": sh(f"{ROOT}/bin/uv --version"),
        "conda": sh(f"{ROOT}/miniconda3/bin/conda --version"),
    }
    sg_src = Path(os.environ.get("SG_SRC", ROOT / "simuguard-src"))  # the formal runs use SG_SRC=simuguard-src-formal
    src_info = sg_src / "SOURCE.json"
    fp["simuguard"] = json.loads(src_info.read_text()) if src_info.exists() else {}
    fp["simuguard"]["synced_tree"] = tree_digest(sg_src / "simuguard", "**/*.py") if sg_src.exists() else None
    fp["simuguard"]["campaign_scripts"] = tree_digest(Path(os.environ.get("SG_SCRIPTS", Path(__file__).resolve().parent)))  # default: the dir this copy runs from
    fp["robodojo"] = git_state(RD)
    py = f"{ROOT}/miniconda3/envs/RoboDojo/bin/python"
    fp["robodojo"]["versions"] = sh(
        f"OMNI_KIT_ACCEPT_EULA=YES {py} -c \"import importlib.metadata as m\n"
        "out=[]\n"
        "for d in ('isaacsim','isaaclab','torch','numpy','warp-lang','nvidia-curobo','omegaconf'):\n"
        "    try: out.append(d+'='+m.version(d))\n"
        "    except Exception: out.append(d+'=MISSING')\n"
        "import torch; out.append('torch_cuda='+str(torch.version.cuda)); print(' '.join(out))\"")
    fp["robodojo"]["freeze"] = freeze("RoboDojo")
    assets = RD / "Assets"
    fp["assets"] = {"listing": listing_digest(assets.resolve())}
    for t in TASKS:
        fp["assets"][f"eval_layout.{t}"] = tree_digest(assets / "Eval_Layout/RoboDojo/arx_x5/0", f"{t}_[0-9]*.json")
    fp["assets"]["robots_curobo"] = tree_digest(assets / "Robots", "*/curobo.yml")
    sums = CKPT / "SHA256SUMS"
    fp["checkpoints"] = {"SHA256SUMS_sha256": sha_file(sums) if sums.exists() else None}
    if sums.exists():
        bad = []
        for line in sums.read_text().splitlines():
            digest, rel = line.split(None, 1)
            path = CKPT / rel.strip()
            if not path.exists():
                bad.append(f"missing {rel}")
            elif args.full and sha_file(path) != digest:
                bad.append(f"sha mismatch {rel}")
        fp["checkpoints"]["check"] = "full sha256" if args.full else "presence only"
        fp["checkpoints"]["problems"] = bad
    base = ROOT / "locks" / "base_models.sha256"
    fp["base_models"] = {"base_models.sha256": sha_file(base) if base.exists() else None}
    tok = ROOT / ".cache/openpi/big_vision/paligemma_tokenizer.model"
    fp["base_models"]["paligemma_tokenizer"] = sha_file(tok) if tok.exists() else None
    fp["policies"] = {}
    for p in args.policies:
        d = POL / p
        entry = {"freeze": freeze(p), "deploy_yml_sha256": sha_file(d / "deploy.yml") if (d / "deploy.yml").exists() else None}
        ck = d / "checkpoints"
        entry["checkpoint_links"] = sorted(
            f"{q.relative_to(ck)} -> {os.readlink(q)}" for q in ck.rglob("*") if q.is_symlink()) if ck.exists() else []
        entry["checkpoint_links_sha256"] = sha_text("\n".join(entry["checkpoint_links"]))
        del entry["checkpoint_links"]
        fp["policies"][p] = entry
    text = json.dumps(fp, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
