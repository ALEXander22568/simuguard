# SAPIEN 3.0.0b1 with the PhysX 5.4.0 GJK simplex threshold (binary patch of a copy), 2026-10-04

The installed venv (`/home/USER/lingbot-va-repro/.venv-client`) is untouched (sha256 of its libsapien.so unchanged).

## Use
```
PYTHONPATH=/mnt/nvme0/shared/USER/simuguard/probe/sapien_patched/site:$PYTHONPATH  <venv python> ...   # patched engine
PYTHONPATH=/mnt/nvme0/shared/USER/simuguard/probe/sapien_patched/site_orig:$PYTHONPATH <venv python> ... # same copy mechanism, unpatched
```
`site/` holds full copies of `sapien/` and `sapien.libs/`; `import sapien` then comes from the copy and pysapien loads
`site/sapien.libs/libsapien.so` through its relative RPATH (check: `sapien.__file__`, `/proc/self/maps`).
`site/sapien.libs/libsapien.so.orig` is the unmodified library; `site_orig/` is a hard linked copy with the unmodified library.
Scripts that prepend their own paths with `sys.path.insert` are fine; a script that sets PYTHONPATH itself must keep `site` first.

## What is patched (patch_lib.py, patch_log.json; 19 bytes differ from the original)
PhysX 5.3.1 `gjk/GuGJKSimplex.h`, both overloads of `closestPtPointTriangle`: `if(FAllGrtrOrEq(eps, area))` with
`area = |ab x ac|^2` and `eps = FEps()` (FLT_EPSILON, 1.19e-7). PhysX 5.4.0 uses `PX_EPS_REAL * PX_EPS_REAL` = 2^-46 = 1.42e-14.
In this build (clang 11) `GJKCPairDoSimplex` and `closestPtPointTetrahedron` are real functions (not inlined into
gjkPenetration / gjk / gjkRaycast, which call them), so the test exists at exactly four places, each
`movss xmmN,[rip+disp] -> FLT_EPSILON at 0x57d900 ; comiss xmmN,xmm_area ; jb`:

| vaddr of movss | function | disp32 at file offset | old | new |
|---|---|---|---|---|
| 0x3e0f07 | physx::Gu::GJKCPairDoSimplex(Vec3V*, Vec3V*, Vec3V*, const Vec3V&, PxU32&) | 0x3e0f0b | f1c91900 | 11681900 |
| 0x3eaec5 | physx::Gu::GJKCPairDoSimplex(Vec3V*, Vec3V*, Vec3V*, PxI32*, PxI32*, const Vec3V&, PxU32&) | 0x3eaec9 | 332a1900 | 53c81800 |
| 0x43f8b0 | physx::Gu::closestPtPointTetrahedron(Vec3V*, Vec3V*, Vec3V*, PxU32&) | 0x43f8b4 | 48e01300 | 687e1300 |
| 0x43ff79 | physx::Gu::closestPtPointTetrahedron(Vec3V*, Vec3V*, Vec3V*, PxI32*, PxI32*, PxU32&) | 0x43ff7d | 7fd91300 | 9f771300 |

The loads now read a new constant, four floats 2^-46 (`00008028` x4) written at file offset = vaddr 0x577720, in the zero
padding between `.fini` (ends 0x577705) and `.rodata` (0x578000); the executable LOAD segment's p_filesz/p_memsz
(program header at file offset 0x98) grow from 0x4ee705 to 0x4ee730 so the constant is inside the mapping.
The shared FLT_EPSILON constants are not modified: the segment test `FIsGrtrOrEq(FEps(), denom)` (cmpleps against
0x5aaf00, 2 per GJKCPairDoSimplex, 1 per closestPtPointTetrahedron) and the eps of closestPtPointTriangleBaryCentric
keep FLT_EPSILON, as in 5.4.0. `scan_eps.py` lists all 114 instructions that reference a FLT_EPSILON constant
(scan.json, scan_ctx.txt); the four above are the only scalar `movss ... comiss` uses inside GJK code, and the 14
gjkPenetration, the gjk and the gjkRaycast instantiations contain no simplex test of their own.
For these four tests the patch is equivalent to 5.4.0 (same value, same comparison). Nothing else of 5.4.0 is included.

## Checks (replay_check.py, run_checks.sh, out/*.json; open loop replay of the recorded actuation, GPU 7)
| run | engine | first step with can position error > 1e-6 m | onset step: deepest can-basket sep / can speed | steps with sep < -5 mm, whole episode | peak can speed, whole episode | success check |
|---|---|---|---|---|---|---|
| Fig. 1 episode (rep2 seg0036, onset 11023, recorded failure) | site_orig | none (max error 0.0 over 60922 steps) | -27.54 mm / 21.807 m/s | 3 | 21.81 m/s | never |
| same | site (patched) | 3499 | -0.008 mm / 0.16 m/s | 0 | 0.81 m/s | true from step 18850 (checked every 25), false at the last recorded step |
| rep1 seg0012 ep5 (onset 11044, recorded failure) | site_orig | none (0.0 over 55822 steps) | -36.53 mm / 2.48 m/s | 2 | 3.70 m/s | never |
| same | site (patched) | 5010 | 0.0 mm / 0.15 m/s | 0 | 0.91 m/s | true from step 19325, true at the end |
| rep1 seg0032 ep14 (onset 11301, recorded success) | site (patched) | 5322 | -0.001 mm / 0.02 m/s | 0 | 0.91 m/s | never (recorded: success) |
| rep1 seg0008 ep3 (onset 12618, recorded success) | site (patched) | 3784 | -0.002 mm / 0.005 m/s | 0 | 0.94 m/s | never (recorded: success) |

## Limits
- The replay leaves the recording thousands of steps before the artifact (3499 to 5322), i.e. the corrected test also changes
  ordinary contacts earlier in the episode (cm sized pieces everywhere), so an open loop replay under the patched engine is
  not a paired counterfactual: the two recorded successes do not reach success, and a rescued failure is only "passes at some
  step with actions recorded for another trajectory". Scores under the patched engine need closed loop runs.
- Only the four sites above are changed; other code paths that 5.4.0 changed (e.g. the hullData1 extent fix) are not.
- The success check was sampled every 25 steps (the official evaluator checks every step).
- The constant sits in segment padding; if a tool rewrites the ELF layout (strip, patchelf) the patch must be redone.
