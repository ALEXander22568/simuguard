#!/usr/bin/env bash
# Run the 13 unit-test fixtures in the Rigid IPC reference solver, as shipped (its own time step 0.01 s), 5 s each.
cd /path/to/workspace/rigid_ipc
for n in 5-cubes edge-edge-parallel edge-edge erleben/cliff-edges erleben/internal-edges erleben/spike-and-wedge erleben/spikes erleben/wedges face-vertex large-mass-ratio tet-corner vertex-face vertex-vertex; do
  o=ripc_out/${n//\//_}; mkdir -p $o
  ( t0=$(date +%s); timeout 3600 ./src/build/rigid_ipc_sim --ngui src/fixtures/3D/unit-tests/$n.json $o --num-steps 500 --nthreads 16 --log 3 > $o/log.txt 2>&1; echo "$n rc=$? $(( $(date +%s)-t0 ))s" >> ripc_out/STATUS ) &
done
wait; echo ALLDONE >> ripc_out/STATUS
