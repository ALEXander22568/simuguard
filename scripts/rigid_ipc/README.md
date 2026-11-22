# Rigid IPC unit tests with SimuGuard attached

The 13 rigid body scenes of the Rigid IPC unit tests, run in MuJoCo, in PhysX and in the
Rigid IPC reference implementation, with SimuGuard recording every step.

`build.sh` builds the reference implementation. `run_mujoco_sg.py`, `run_physx_sg.py` and
`run_ripc_sg.py` run the scenes in each simulator and write one SimuGuard record per scene.
`common.py` holds the scene loader and the speed check. `results/summary.md` is the table
of the runs.
