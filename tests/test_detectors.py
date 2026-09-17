import unittest

import numpy as np

from simuguard.core.detectors import (
    ActuationBoundDetector,
    ContactEjectionDetector,
    ImpulseSpikeDetector,
    NonFiniteStateDetector,
    PenetrationDetector,
)
from simuguard.core.detectors.base import DetectorContext
from simuguard.core.events import EventStatus
from simuguard.core.types import BodyInfo, BodyKind, BodyRole, BodyState, ContactPair, ContactPoint, SubstepFrame

DT = 0.004
Q = np.array([1.0, 0.0, 0.0, 0.0])
BODIES = {
    "can": BodyInfo("can", "can", BodyKind.DYNAMIC, BodyRole.TARGET, mass=0.01),
    "basket": BodyInfo("basket", "basket", BodyKind.DYNAMIC, BodyRole.CONTAINER, mass=0.5),
    "table": BodyInfo("table", "table", BodyKind.STATIC, BodyRole.SCENE),
    "gripper": BodyInfo("gripper", "gripper", BodyKind.LINK, BodyRole.ROBOT, mass=0.1),
}


def context():
    return DetectorContext(episode_id="ep", timestep=DT, bodies=BODIES)


def frame(i, can_p, can_v, contacts=(), gripper_v=(0, 0, 0)):
    states = {
        "can": BodyState("can", np.asarray(can_p, float), Q, np.asarray(can_v, float), np.zeros(3)),
        "basket": BodyState("basket", np.zeros(3), Q, np.zeros(3), np.zeros(3)),
        "gripper": BodyState("gripper", np.array([0.3, 0, 0.2]), Q, np.asarray(gripper_v, float), np.zeros(3)),
    }
    return SubstepFrame(i, i * DT, i // 10, DT, states, list(contacts))


def contact(a="can", b="basket", impulse=(0, 0, 1e-4), separation=-1e-4):
    return ContactPair(a, b, [ContactPoint(np.zeros(3), np.array([0, 0, 1.0]), np.asarray(impulse, float), separation)])


def run(detector, frames):
    ctx = context()
    detector.reset(ctx)
    events = {}
    for f in frames:
        for e in detector.observe(f, ctx):
            events[e.event_id] = e
    for e in detector.finalize(frames[-1] if frames else None, ctx):
        events[e.event_id] = e
    return list(events.values())


class EjectionTests(unittest.TestCase):
    def test_resting_contact_has_no_events(self):
        frames = [frame(i, (0, 0, 0), (0, 0, 0.001 * (-1) ** i), [contact()]) for i in range(1, 300)]
        self.assertEqual(run(ContactEjectionDetector(), frames), [])

    def test_ballistic_ejection_confirmed(self):
        frames = [frame(i, (0, 0, 0), (0, 0, 0), [contact()]) for i in range(1, 50)]
        p = np.zeros(3)
        v = np.array([0.0, 0.0, 3.0])
        for i in range(50, 200):
            v = v + np.array([0, 0, -9.81]) * DT
            p = p + v * DT
            frames.append(frame(i, p.copy(), v.copy(), [contact()] if i == 50 else []))
        events = run(ContactEjectionDetector(), frames)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].status, EventStatus.CONFIRMED)
        self.assertIn("ballistic_free_flight", events[0].reasons)
        self.assertTrue(events[0].metrics["risky_contact"])

    def test_violent_path_with_persistent_contact(self):
        frames = [frame(i, (0, 0, 0), (0, 0, 0), [contact()]) for i in range(1, 50)]
        for k, i in enumerate(range(50, 120)):
            frames.append(frame(i, (0.0, 1.5 * (k + 1) * DT, 0.0), (0.0, 1.5, 0.0), [contact()]))
        events = run(ContactEjectionDetector(), frames)
        self.assertEqual([e.status for e in events], [EventStatus.CONFIRMED])
        self.assertEqual(events[0].reasons, ["violent_risky_contact_displacement"])

    def test_spike_without_displacement_is_rejected(self):
        frames = [frame(i, (0, 0, 0), (0, 0, 0), [contact()]) for i in range(1, 50)]
        frames.append(frame(50, (0, 0, 0), (0, 0, 0.6), [contact()]))
        frames += [frame(i, (0, 0, 0), (0, 0, 0), [contact()]) for i in range(51, 200)]
        events = run(ContactEjectionDetector(), frames)
        self.assertEqual([e.status for e in events], [EventStatus.REJECTED])

    def test_gate_result_is_attached(self):
        calls = []

        def gate(f, body_id, ctx):
            calls.append(body_id)
            return {"active": True}

        frames = [frame(i, (0, 0, 0), (0, 0, 0), [contact()]) for i in range(1, 50)]
        frames.append(frame(50, (0, 0, 0.01), (0, 0, 3.0), [contact()]))
        events = run(ContactEjectionDetector(gate=gate), frames)
        self.assertEqual(calls, ["can"])
        self.assertEqual(events[0].metrics["gate_at_onset"], {"active": True})


class SignalTests(unittest.TestCase):
    def test_penetration_flag_with_cooldown(self):
        frames = [frame(i, (0, 0, 0), (0, 0, 0), [contact(separation=-0.01)]) for i in range(1, 30)]
        events = run(PenetrationDetector(), frames)
        self.assertEqual(len(events), 2)  # cooldown 100 ms = 25 substeps
        self.assertTrue(all(e.status == EventStatus.FLAG for e in events))

    def test_impulse_spike(self):
        frames = [frame(1, (0, 0, 0), (0, 0, 0), [contact(impulse=(0, 0, 2.0))])]
        events = run(ImpulseSpikeDetector(), frames)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].metrics["force_estimate_n"], 500.0)

    def test_non_finite_once_per_body(self):
        frames = [frame(i, (0, 0, np.nan), (0, 0, 0)) for i in range(1, 5)]
        events = run(NonFiniteStateDetector(), frames)
        self.assertEqual(len(events), 1)

    def test_actuation_bound(self):
        frames = [frame(i, (0, 0, 0), (0, 0, 0), [contact()], gripper_v=(0.2, 0, 0)) for i in range(1, 10)]
        frames.append(frame(10, (0, 0, 0.01), (0, 0, 4.0), [contact()], gripper_v=(0.2, 0, 0)))
        events = run(ActuationBoundDetector(), frames)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].metrics["limit_mps"], 3 * 0.2 + 0.5)


if __name__ == "__main__":
    unittest.main()
