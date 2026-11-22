"""Pure parts of scripts/repack_daemon.py and scripts/mirror_daemon.py (LZMA repack, robot files kept deflate, sha256 compare)."""

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TMP = Path(tempfile.mkdtemp())
os.environ["REPACK_RUNS"] = str(TMP)
os.environ["MIRROR_RUNS"] = str(TMP)


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


repack = load("repack_daemon")
mirror = load("mirror_daemon")


def make_task(name: str) -> Path:
    task = TMP / "formal_pi05" / name
    batch = task / name / "batch"
    batch.mkdir(parents=True)
    (TMP / "formal_pi05" / "_claims").mkdir(exist_ok=True)
    (TMP / "formal_pi05" / "_claims" / f"{name}.lock").touch()
    rng = np.random.default_rng(0)
    np.savez_compressed(task / name / "contacts.npz", a=rng.normal(size=(50, 3)).astype(np.float32), n=np.arange(7))
    np.savez_compressed(batch / "run_0_robot.npz", dof_pos=rng.normal(size=(20, 2, 13)).astype(np.float32))
    (task / "DONE.json").write_text(json.dumps({"ok": True}))
    return task


def kinds(path: Path) -> set[int]:
    with zipfile.ZipFile(path) as z:
        return {i.compress_type for i in z.infolist()}


class RepackTest(unittest.TestCase):
    def test_repack_lzma_then_back_to_deflate(self):
        task = make_task("A")
        path = task / "A" / "contacts.npz"
        orig = path.read_bytes()
        row = repack.repack(str(path))
        self.assertEqual(row["verify"], "identical")
        self.assertEqual(row["sha256_orig"], hashlib.sha256(orig).hexdigest())
        self.assertEqual(kinds(path), {zipfile.ZIP_LZMA})
        with np.load(path) as d:
            np.testing.assert_array_equal(d["n"], np.arange(7))
        back = repack.repack(str(path), zipfile.ZIP_DEFLATED)
        self.assertEqual((back["to"], back["verify"]), ("deflate", "identical"))
        self.assertEqual(kinds(path), {zipfile.ZIP_DEFLATED})
        self.assertFalse(list(path.parent.glob(".*.repack-tmp")))

    def test_candidates_skip_robot_files(self):
        task = make_task("B")
        found = {t.name: [f.name for f in files] for _, t, files in repack.candidates()}
        self.assertEqual(found["B"], ["contacts.npz"])
        self.assertFalse(mirror.repacked(task))
        repack.repack(str(task / "B" / "contacts.npz"))
        self.assertNotIn("B", {t.name for _, t, _ in repack.candidates()})
        self.assertTrue(mirror.repacked(task))

    def test_differing_arrays_are_reported(self):
        a, b = TMP / "x.npz", TMP / "y.npz"
        np.savez(a, v=np.zeros(3))
        np.savez(b, v=np.ones(3))
        self.assertEqual(repack.same_arrays(a, b), "array v differs")


class MirrorTest(unittest.TestCase):
    def test_hashes_match_sha256sum_output(self):
        task = make_task("C")
        (task / "C" / ".contacts.npz.repack-tmp").write_bytes(b"x")
        local = mirror.local_hashes(task)
        self.assertNotIn("C/.contacts.npz.repack-tmp", local)
        text = "".join(f"{v}  ./{k}\n" for k, v in local.items())
        self.assertEqual(mirror.parse_sha256sum(text), local)
        self.assertEqual(mirror.compare(local, mirror.parse_sha256sum(text)),
                         {"missing_remote": [], "extra_remote": [], "mismatched": []})

    def test_compare_finds_differences(self):
        diff = mirror.compare({"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "9", "d": "4"})
        self.assertEqual(diff, {"missing_remote": ["c"], "extra_remote": ["d"], "mismatched": ["b"]})

    def test_mirrored_reads_only_matching_rows(self):
        log = TMP / "MIRROR.jsonl"
        log.write_text(json.dumps({"task": "A", "match": True}) + "\n" + json.dumps({"task": "B", "match": False}) + "\n")
        self.assertEqual(mirror.mirrored(log), {"A"})


if __name__ == "__main__":
    unittest.main()
