Rigid IPC unit tests (13 scenes from comparisons/MuJoCo/unit-tests), 5 s each, zero friction, convex hulls, density 1000.
excess = largest ratio of a body speed to the free-fall speed from the highest starting point in the scene; kept = confirmed ejection events with gravity ratio > 1.5.

| run | scenes | scenes with excess > 1.5 | scenes flagged | flagged and excess | flagged, no excess | excess, not flagged |
|---|---|---|---|---|---|---|
| MuJoCo, 2 ms | 13 | 0 | 0 | 0 | 0 | 0 |
| MuJoCo, 4 ms | 13 | 0 | 0 | 0 | 0 | 0 |
| PhysX, 4 ms | 13 | 1 | 2 | 1 | 1 | 0 |
| PhysX, 2 ms | 13 | 1 | 1 | 1 | 0 | 0 |
| PhysX smaller threshold, 4 ms | 13 | 1 | 2 | 1 | 1 | 0 |
| PhysX smaller threshold, 2 ms | 13 | 1 | 1 | 1 | 0 | 0 |

| scene | MuJoCo, 2 ms | MuJoCo, 4 ms | PhysX, 4 ms | PhysX, 2 ms |
|---|---|---|---|---|
| 5-cubes | peak 10.38 m/s, excess 1.00, kept 0 | peak 10.40 m/s, excess 1.00, kept 0 | peak 10.20 m/s, excess 1.00, kept 1 | peak 10.24 m/s, excess 1.00, kept 0 |
| edge-edge-parallel | peak 44.00 m/s, excess 1.00, kept 0 | peak 43.97 m/s, excess 1.00, kept 0 | peak 44.12 m/s, excess 1.00, kept 0 | peak 44.09 m/s, excess 1.00, kept 0 |
| edge-edge | peak 2.39 m/s, excess 1.00, kept 0 | peak 3.84 m/s, excess 0.99, kept 0 | peak 2.35 m/s, excess 0.99, kept 0 | peak 2.37 m/s, excess 1.00, kept 0 |
| erleben/cliff-edges | peak 0.98 m/s, excess 0.99, kept 0 | peak 0.98 m/s, excess 0.98, kept 0 | peak 0.94 m/s, excess 0.98, kept 0 | peak 0.96 m/s, excess 0.99, kept 0 |
| erleben/internal-edges | peak 0.98 m/s, excess 0.99, kept 0 | peak 0.98 m/s, excess 0.98, kept 0 | peak 0.94 m/s, excess 0.98, kept 0 | peak 0.96 m/s, excess 0.99, kept 0 |
| erleben/spike-and-wedge | peak 43.68 m/s, excess 1.00, kept 0 | peak 43.64 m/s, excess 1.00, kept 0 | peak 43.68 m/s, excess 1.00, kept 0 | peak 43.62 m/s, excess 1.00, kept 0 |
| erleben/spikes | peak 43.67 m/s, excess 1.00, kept 0 | peak 43.62 m/s, excess 1.00, kept 0 | peak 43.68 m/s, excess 1.00, kept 0 | peak 43.62 m/s, excess 1.00, kept 0 |
| erleben/wedges | peak 45.54 m/s, excess 1.00, kept 0 | peak 45.51 m/s, excess 1.00, kept 0 | peak 45.54 m/s, excess 1.00, kept 0 | peak 45.51 m/s, excess 1.00, kept 0 |
| face-vertex | peak 44.02 m/s, excess 1.00, kept 0 | peak 44.01 m/s, excess 1.00, kept 0 | peak 44.08 m/s, excess 1.00, kept 0 | peak 44.09 m/s, excess 1.00, kept 0 |
| large-mass-ratio | peak 3.14 m/s, excess 1.00, kept 0 | peak 3.14 m/s, excess 0.99, kept 0 | peak 66.83 m/s, excess 5.45, kept 4 | peak 63.24 m/s, excess 4.73, kept 4 |
| tet-corner | peak 4.65 m/s, excess 1.00, kept 0 | peak 4.63 m/s, excess 1.00, kept 0 | peak 2.57 m/s, excess 0.99, kept 0 | peak 3.00 m/s, excess 0.99, kept 0 |
| vertex-face | peak 2.39 m/s, excess 1.00, kept 0 | peak 2.39 m/s, excess 0.99, kept 0 | peak 2.35 m/s, excess 0.99, kept 0 | peak 2.37 m/s, excess 1.00, kept 0 |
| vertex-vertex | peak 20.97 m/s, excess 1.00, kept 0 | peak 20.98 m/s, excess 0.99, kept 0 | peak 10.83 m/s, excess 0.99, kept 0 | peak 14.72 m/s, excess 1.00, kept 0 |
