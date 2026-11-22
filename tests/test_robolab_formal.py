"""Pure parts of scripts/robolab_formal.py (task list, completion checks, timeout)."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("robolab_formal", Path(__file__).resolve().parents[1] / "scripts" / "robolab_formal.py")
formal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formal)


def make_task(root: Path, runs: int = 2, **run_fields) -> Path:
    task_dir = root / "T"
    batch = task_dir / "T" / "batch"
    batch.mkdir(parents=True)
    (batch / "install.json").write_text(json.dumps({"contact_view_errors": {}}))
    for k in range(runs):
        run = {"unknown_views": {}, "apply_force_calls": 0, "segments": [{"monitor_errors": 0}], **run_fields}
        (batch / f"run_{k}.json").write_text(json.dumps(run))
        (batch / f"run_{k}_writes.npz").write_bytes(b"x")
        (batch / f"run_{k}_robot.npz").write_bytes(b"x")
    with (task_dir / "run_index.jsonl").open("w") as fh:
        for k in range(runs):
            fh.write(json.dumps({"run": k}) + "\n")
    return task_dir


class FormalTests(unittest.TestCase):
    def test_benchmark_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp) / "robolab" / "tasks" / "benchmark"
            bench.mkdir(parents=True)
            (bench / "a.py").write_text("class FooTask(Task):\n    pass\nclass Helper:\n    pass\n")
            (bench / "b.py").write_text("class BarTask(Task):\n    pass\n")
            self.assertEqual(formal.benchmark_tasks(Path(tmp)), ["BarTask", "FooTask"])

    def test_clean_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = make_task(Path(tmp))
            self.assertEqual(formal.runs_done(task_dir), 2)
            self.assertEqual(formal.task_issues(task_dir, 2), [])

    def test_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = make_task(Path(tmp), runs=1, apply_force_calls=3, segments=[{"monitor_errors": 2}])
            issues = formal.task_issues(task_dir, 2)
            self.assertIn("run 0: 3 apply_* calls", issues)
            self.assertIn("run 0: 2 monitor errors", issues)
            self.assertIn("missing run_1.json", issues)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(formal.task_issues(Path(tmp), 1), ["no batch directory"])

    def test_timeout_kills(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log"
            t0 = time.time()
            code = formal.run_task([sys.executable, "-c", "import time; time.sleep(30)"], tmp, dict(os.environ), log, 1.0)
            self.assertIsNone(code)
            self.assertLess(time.time() - t0, 10)
            self.assertEqual(formal.run_task([sys.executable, "-c", "print(1)"], tmp, dict(os.environ), log, 10.0), 0)
            self.assertEqual(log.read_text().strip(), "1")

    def test_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            fd = formal.claim(out, "A")
            self.assertIsNotNone(fd)
            self.assertIsNone(formal.claim(out, "A"))
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], pass_fds=(fd,))
            os.close(fd)
            self.assertIsNone(formal.claim(out, "A"))
            child.kill()
            child.wait()
            fd = formal.claim(out, "A")
            self.assertIsNotNone(fd)
            self.assertEqual(json.loads((out / "_claims" / "A.lock").read_text())["pid"], os.getpid())
            os.close(fd)

    def test_driver_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "robolab" / ".venv" / "bin" / "python"
            fake.parent.mkdir(parents=True)
            fake.write_text(
                "#!/bin/sh\n"
                "while [ $# -gt 0 ]; do case $1 in --task) t=$2;; --out) o=$2;; esac; shift; done\n"
                "[ $t = Bad ] && exit 3\n"
                "mkdir -p $o/$t/batch; echo '{}' > $o/$t/batch/install.json\n"
                "for k in 0 1; do echo '{\"segments\": []}' > $o/$t/batch/run_$k.json; "
                "touch $o/$t/batch/run_${k}_writes.npz $o/$t/batch/run_${k}_robot.npz; echo \"{\\\"run\\\": $k}\" >> $o/run_index.jsonl; done\n")
            fake.chmod(0o755)
            out = root / "out"
            argv = ["robolab_formal.py", "--policy", "gr00t", "--gpu", "0", "--port", "1", "--out", str(out),
                    "--robolab", str(root / "robolab"), "--tasks", "Good", "Bad", "--attempts", "2"]
            old = sys.argv
            sys.argv = argv
            try:
                self.assertEqual(formal.main(), 1)
                self.assertEqual(formal.main(), 1)
            finally:
                sys.argv = old
            done = json.loads((out / "Good" / "DONE.json").read_text())
            self.assertEqual((done["ok"], done["issues"]), (True, []))
            self.assertEqual(formal.failures(out, "Bad"), 2)
            self.assertFalse((out / "Bad").exists())
            self.assertTrue(all(formal.claim(out, t) is not None for t in ("Good", "Bad")))
            rows = [json.loads(line) for line in next(out.glob("driver_*.jsonl")).read_text().splitlines()]
            self.assertEqual([r["task"] for r in rows if r["event"] == "task"], ["Good", "Bad", "Bad"])
            self.assertEqual(rows[0]["args"]["open_loop_horizon"], 8)


if __name__ == "__main__":
    unittest.main()
