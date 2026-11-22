"""scripts/remote_claim.py and the cross-machine parts of scripts/robolab_formal.py."""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = load("remote_claim")
formal = load("robolab_formal")


def wait_gone(pid, seconds=10.0):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if not rc.alive(pid):
            return True
        time.sleep(0.05)
    return False


class RemoteClaimTests(unittest.TestCase):
    def test_parse_request(self):
        out = "/path/to/simuguard-robolab/runs/formal_gr00t"
        self.assertEqual(rc.parse_request(["acquire", out, "FooTask", "48g-gpu2"]), ["acquire", out, "FooTask", "48g-gpu2"])
        self.assertEqual(rc.parse_request(["acquire", out, "FooTask", "48g", "3"])[4], "3")
        self.assertEqual(rc.parse_request(["release", out, "FooTask", "48g", "done"])[4], "done")
        self.assertEqual(rc.parse_request(["status", out, "FooTask"])[0], "status")
        for bad in ([], ["rm", out, "T", "o"], ["acquire", "/root", "T", "o"], ["acquire", out, "T;rm", "o"],
                    ["acquire", out, "../T", "o"], ["release", out, "T", "o", "maybe"], ["acquire", out, "T", "o", "x"],
                    ["status", out, "T", "extra"]):
            with self.assertRaises(ValueError, msg=bad):
                rc.parse_request(bad)

    def test_ssh_original_command(self):
        os.environ["SSH_ORIGINAL_COMMAND"] = "rm -rf /"
        try:
            self.assertEqual(rc.main([]), 3)
        finally:
            del os.environ["SSH_ORIGINAL_COMMAND"]

    def test_acquire_blocks_local_drivers_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(rc.acquire(out, "A", "48g", poll=0.05), "claimed")
            mark = rc.marker(out, "A")
            pid = rc.holder_pid(mark)
            self.assertTrue(rc.alive(pid))
            self.assertEqual(json.loads((mark / "owner.json").read_text())["owner"], "48g")
            self.assertIsNone(formal.claim(out, "A"))
            self.assertTrue(formal.remote_taken(out, "A"))
            self.assertEqual(rc.acquire(out, "A", "other", poll=0.05), "busy")
            self.assertEqual(rc.acquire(out, "A", "48g", poll=0.05), "claimed")
            self.assertEqual(rc.status(out, "A"), "busy")
            self.assertEqual(rc.release(out, "A", "other", "failed"), "busy")
            self.assertEqual(rc.release(out, "A", "48g", "failed"), "released")
            self.assertTrue(wait_gone(pid))
            self.assertFalse(formal.remote_taken(out, "A"))
            self.assertEqual(rc.status(out, "A"), "free")
            fd = formal.claim(out, "A")
            self.assertIsNotNone(fd)
            self.assertEqual(rc.acquire(out, "A", "48g", poll=0.05), "busy")
            os.close(fd)

    def test_done_keeps_holder(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(rc.acquire(out, "A", "48g", poll=0.05), "claimed")
            mark = rc.marker(out, "A")
            pid = rc.holder_pid(mark)
            self.assertEqual(rc.release(out, "A", "48g", "done"), "released")
            time.sleep(0.2)
            self.assertTrue(rc.alive(pid))
            self.assertIsNone(formal.claim(out, "A"))
            self.assertEqual(rc.acquire(out, "A", "48g", poll=0.05), "done")
            self.assertEqual(rc.status(out, "A"), "done")
            shutil.rmtree(mark)
            self.assertTrue(wait_gone(pid))

    def test_dead_holder_reacquired_by_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(rc.acquire(out, "A", "48g", poll=0.05), "claimed")
            mark = rc.marker(out, "A")
            pid = rc.holder_pid(mark)
            os.kill(pid, 9)
            self.assertTrue(wait_gone(pid))
            self.assertEqual(rc.acquire(out, "A", "other", poll=0.05), "busy")
            self.assertEqual(rc.acquire(out, "A", "48g", poll=0.05), "claimed")
            pid2 = rc.holder_pid(mark)
            self.assertNotEqual(pid, pid2)
            self.assertIsNone(formal.claim(out, "A"))
            rc.release(out, "A", "48g", "failed")
            self.assertTrue(wait_gone(pid2))

    def test_done_or_failed_here(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "A").mkdir()
            (out / "A" / "DONE.json").write_text("{}")
            self.assertEqual(rc.acquire(out, "A", "48g"), "done")
            (out / "_failed").mkdir()
            for k in range(3):
                (out / "_failed" / f"B.{k}").mkdir()
            self.assertEqual(rc.acquire(out, "B", "48g", attempts=3), "done")
            self.assertEqual(rc.acquire(out, "B", "48g", attempts=4, poll=0.05), "claimed")
            rc.release(out, "B", "48g", "failed")

    def test_remote_call(self):
        self.assertEqual(formal.remote_call(["sh", "-c", "echo claimed; exit 0", "sh"], ["acquire", "x"]), "claimed")
        self.assertEqual(formal.remote_call(["sh", "-c", "echo busy; exit 1", "sh"], ["acquire", "x"]), "busy")
        self.assertIsNone(formal.remote_call(["sh", "-c", "exit 255", "sh"], ["acquire", "x"], tries=2, wait=0.01))
        self.assertEqual(formal.remote_call([sys.executable, "-c", "import sys; print(sys.argv[1].split()[0])"], ["status", "o", "T"]), "status")


if __name__ == "__main__":
    unittest.main()
