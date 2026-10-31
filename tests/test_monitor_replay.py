import gzip
import json
import tempfile
import unittest
from pathlib import Path

from simuguard.core.detectors import ContactEjectionDetector, Detector
from simuguard.core.events import EventStatus
from simuguard.core.monitor import MonitorConfig, SubstepMonitor
from simuguard.core.recorder import EpisodeRecorder
from simuguard.core.replay import replay_bundle
from simuguard.core.snapshot import ReplayBundle
from simuguard.presets import default_detectors

from fakes import ToyAdapter, run_kick_episode


class ExplodingDetector(Detector):
    name = "exploding"

    def observe(self, frame, context):
        raise RuntimeError("boom")


class MonitorTests(unittest.TestCase):
    def _monitored_run(self, tmp, detectors, **cfg):
        adapter = ToyAdapter()
        monitor = SubstepMonitor(
            adapter,
            detectors,
            episode_id="toy-ep",
            config=MonitorConfig(snapshot_interval_substeps=20, bundle_post_substeps=60, bundle_mode="snapshot", **cfg),
            recorder=EpisodeRecorder(tmp),
        )
        monitor.attach()
        run_kick_episode(adapter)
        return adapter, monitor, monitor.finalize()

    def test_confirmed_event_bundle_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, monitor, summary = self._monitored_run(tmp, default_detectors())
            self.assertEqual(summary["substeps"], 200)
            self.assertEqual(summary["confirmed_count"], 1)
            self.assertEqual(summary["error_count"], 0, summary["errors"])
            [event] = [e for e in summary["events"] if e["status"] == "confirmed"]
            self.assertEqual(event["onset_substep"], 80)
            bundle_path = summary["bundles"][event["event_id"]]
            bundle = ReplayBundle.load(bundle_path)
            self.assertEqual(bundle.start_substep, 80)
            self.assertEqual(bundle.end_substep, 140)
            root = Path(tmp)
            for name in ("manifest.json", "summary.json", "events.jsonl", "trace.jsonl.gz"):
                self.assertTrue((root / name).exists(), name)
            with gzip.open(root / "trace.jsonl.gz", "rt") as handle:
                first = json.loads(handle.readline())
            self.assertNotIn("link:robot:gripper", first["states"])  # robot excluded from trace
            self.assertEqual(monitor.finalize(), summary)  # idempotent

    def test_detector_failure_never_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, summary = self._monitored_run(tmp, [ExplodingDetector(), ContactEjectionDetector()])
            self.assertEqual(summary["substeps"], 200)
            self.assertEqual(summary["error_count"], 200)
            self.assertEqual(summary["confirmed_count"], 1)

    def test_hook_removed_after_finalize(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter, monitor, _ = self._monitored_run(tmp, [])
            self.assertEqual(adapter._callbacks, [])
            self.assertFalse(monitor.attached)


class ReplayTests(unittest.TestCase):
    def _bundle(self, tmp):
        adapter = ToyAdapter()
        monitor = SubstepMonitor(
            adapter,
            [ContactEjectionDetector()],
            episode_id="toy-ep",
            config=MonitorConfig(snapshot_interval_substeps=20, bundle_post_substeps=60, bundle_mode="snapshot"),
            recorder=EpisodeRecorder(tmp),
        )
        monitor.attach()
        # snapshots at multiples of 20 -> kick at 70 means replay starts from 60 and must replay the kick
        run_kick_episode(adapter, kick_at=70)
        summary = monitor.finalize()
        [path] = summary["bundles"].values()
        return adapter, ReplayBundle.load(path)

    def test_replay_reproduces_trajectory_and_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter, bundle = self._bundle(tmp)
            self.assertEqual(bundle.start_substep, 60)
            for method in ("native", "public"):
                result = replay_bundle(adapter, bundle, method=method, detectors=[ContactEjectionDetector()])
                self.assertTrue(result.restore["ok"])
                self.assertEqual(result.restore["public_state_error"]["max_abs_error"], 0.0)
                self.assertTrue(result.within_tolerance, result.to_dict())
                self.assertLess(result.overall_max_error_m, 1e-12)
                self.assertEqual(len(result.confirmed_events), 1)

    def test_intervention_changes_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter, bundle = self._bundle(tmp)

            def remove_kick(a):
                original = a.apply_control

                def no_kick(control):
                    original(control)
                    a.command["kick"] = [0.0, 0.0, 0.0]

                a.apply_control = no_kick

            result = replay_bundle(adapter, bundle, detectors=[ContactEjectionDetector()], before_replay=remove_kick)
            self.assertFalse(result.within_tolerance)
            self.assertEqual(result.confirmed_events, [])


class EpisodeStartBundleTests(unittest.TestCase):
    def test_episode_start_bundle_replays_on_rebuilt_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = ToyAdapter()
            monitor = SubstepMonitor(
                adapter,
                [ContactEjectionDetector()],
                episode_id="toy-ep",
                config=MonitorConfig(snapshot_interval_substeps=20, snapshot_capacity=2, bundle_post_substeps=60),
                recorder=EpisodeRecorder(tmp),
            )
            monitor.attach()
            run_kick_episode(adapter, kick_at=150)
            summary = monitor.finalize()
            [path] = summary["bundles"].values()
            bundle = ReplayBundle.load(path)
            self.assertEqual(bundle.start_substep, 0)  # pinned initial snapshot survives capacity=2
            self.assertEqual(bundle.end_substep, 200)
            rebuilt = ToyAdapter()  # deterministic rebuild, restore method "none"-equivalent: public state already equal
            result = replay_bundle(rebuilt, bundle, method="public", detectors=[ContactEjectionDetector()])
            self.assertTrue(result.within_tolerance)
            self.assertEqual(len(result.confirmed_events), 1)


class StatusSemanticsTests(unittest.TestCase):
    def test_event_status_values(self):
        self.assertEqual(EventStatus.CONFIRMED.value, "confirmed")


if __name__ == "__main__":
    unittest.main()
