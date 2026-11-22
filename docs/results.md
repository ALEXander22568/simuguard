# Results

Numbers are those of the paper. Each section names the files that hold the data or the code.

## Five models against a real robot

Five models run the five container tasks of RoboTwin 2.0, 20 episodes per task, in the
released engine and in the repaired engine. The same models run on a real dual-arm Piper-X
robot: three tasks, 10 trials each, after fine-tuning on 100 real demonstrations per task.

| Model | RoboTwin | Repaired | Real robot |
|---|---|---|---|
| LingBot-VA | 89/100 | 98/100 | 24/30 |
| InternVLA-A1 | 86/100 | 98/100 | 23/30 |
| HyVLA | 90/100 | 94/100 | 20/30 |
| LingBot-VLA | 82/100 | 95/100 | 21/30 |
| StarVLA | 60/100 | 71/100 | 10/30 |

As released, RoboTwin ranks HyVLA first, although it is fourth on the robot. In the repaired
engine the five models fall into the same order as on the robot. Spearman's correlation with
the real ranking is 0.40 as released and 0.97 in the repaired engine.

Data: [`results/five_models_vs_real_robot.csv`](results/five_models_vs_real_robot.csv),
[`results/five_models_per_task.csv`](results/five_models_per_task.csv).

## Every benchmark

Success rate in percent, as reported and after removing episodes with a real artifact.

| Benchmark | Engine | Policy | Episodes | Reported | Corrected |
|---|---|---|---|---|---|
| RoboTwin 2.0 | PhysX, SAPIEN | LingBot-VA | 2060 | 90.8 | 93.0 |
| RoboTwin 2.0 | PhysX, SAPIEN | HyVLA | 220 | 78.2 | 80.0 |
| RoboLab | PhysX, Isaac Sim | π0.5-DROID | 2400 | 28.1 | 28.5 |
| RoboLab | PhysX, Isaac Sim | GR00T N1.7-DROID | 2400 | 8.8 | 9.1 |
| RoboDojo | PhysX, Isaac Sim | XR-1 | 40 | 40.0 | 48.5 |
| RoboDojo | PhysX, Isaac Sim | InternW0 | 360 | 39.7 | 40.6 |
| RoboDojo | PhysX, Isaac Sim | OpenWAM | 360 | 19.4 | 19.8 |
| RoboDojo | PhysX, Isaac Sim | π0.5 | 360 | 17.1 | 17.0 |
| RoboDojo | PhysX, Isaac Sim | X-VLA | 360 | 15.8 | 16.7 |
| RoboDojo | PhysX, Isaac Sim | Xiaomi-Robotics-0 | 360 | 5.2 | 5.3 |
| RoboDojo | PhysX, Isaac Sim | π0 | 360 | 3.6 | 3.7 |
| RoboDojo | PhysX, Isaac Sim | InternVLA-A1 | 360 | 2.1 | 2.1 |
| RoboDojo | PhysX, Isaac Sim | GR00T N1.7 | 360 | 1.3 | 1.3 |
| LIBERO | MuJoCo | π0.5 | 400 | 95.2 | 95.5 |
| RoboCasa | MuJoCo | XR-1 | 300 | 61.7 | 62.7 |

Code: `simuguard/integrations/robolab_eval.py` and `scripts/robolab_formal.py` for RoboLab,
`scripts/robodojo/campaign/` for RoboDojo, `scripts/libero/` and `scripts/robocasa/` for MuJoCo.

## Two other simulation errors

**Objects that start inside their support.** On LIBERO, 650 of the 2000 stored initial states
place objects inside the floor or the table, up to 37.7 mm deep. MuJoCo pushes them out at up
to 1.83 m/s before the policy acts. See [`libero.md`](libero.md) and
`scripts/libero/init_state_audit.py`.

**Objects stuck to an open gripper.** On RoboLab, a fixed rule flags 179 of the 4800 episodes.
See [`scripts/robolab/sticking/`](../scripts/robolab/sticking/README.md).

## Short limit on one body, and a control

On RoboTwin, the depenetration limit is applied to one body for a short window around the
artifact, and the same change is applied to matched episodes without an artifact as a control.
Plan and results: `scripts/robotwin/narrow/`.

## Rigid IPC unit tests

The 13 scenes of the Rigid IPC unit tests in three simulators with SimuGuard attached:
[`scripts/rigid_ipc/`](../scripts/rigid_ipc/README.md).
