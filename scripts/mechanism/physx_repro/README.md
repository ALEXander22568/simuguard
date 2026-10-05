# PhysX 5.3.1 standalone reproduction of the inverted contact normal (2026-10-04)

Build: `./build.sh` (g++ only; foundation + common/geomutils + cooking -> build/libphysx_min.a, build/repro).
Debug prints: `python3 sg_patch.py && python3 sg_patch2.py` (restores from *.orig first), then
`./build.sh 'GuPCMContactConvexConvex|GuPCMContactGenBoxConvex'`. Originals are kept as *.orig.

Data: `export_pair.py` (bit-exact SAPIEN replay; hull vertices, scale, local pose, actor pose before every step,
SAPIEN's contacts for the pair). `data/fig1` (rep2 seg0036, onset 11023, can piece 11 / basket piece 10),
`data/ep5` (rep1 seg0012, 11044, 17/46), `data/ep14` (rep1 seg0032, 11301, 19/18), `data/ep3` (rep1 seg0008, 12618, 16/41).

Run: `build/repro DATA_DIR [order=0] [toleranceLength=0.1] [contactDist=0.02] [startStep=1] [verbose=0] [noBroadphaseReset=0]`
env `SG_CLEAR_WARM_AT=<step>` / `SG_CLEAR_WARM_ALWAYS=1` zero the cached GJK warm start.
The program calls Gu::pcmContactConvexConvex with one persistent cache; the cache is cleared only when the
inflated bounds of the two pieces stop overlapping (as the broadphase does).

Result: all four episodes reproduce SAPIEN's onset contact to the printed digits (normal and separations).
Path at the onset step (same in all four):
1. gjkPenetration (GuGJKPenetration.h) leaves the loop through `bNotDegenerated = FIsGrtr(prevDist, dist)` = false
   with a 2 point simplex: GJK_DEGENERATE, closest points of the unconverged simplex, no test against contactDist.
2. addGJKEPAContacts (GuPCMContactGenBoxConvex.cpp) accepts it (cos(searchDir, normal) > 0.9999 and centre direction
   within 45 degrees): one manifold point, doOverlapTest = false.
3. generateOrProcessContactsConvexConvex (GuPCMContactConvexConvex.cpp): fullContactGen = true.
4. generateFullContactManifold, branch `doOverlapTest == false`: witness faces from the GJK closest points.
   getWitnessPolygonIndex (GuPCMContactGenUtil.cpp) returns a face of shape 0 (can piece) whose normal points AWAY
   from shape 1 (signed dot with the GJK normal +0.68 to +0.99); incidentProject > referenceProject makes that face
   the reference; the contact normal is set to minus that face normal (`nn = V3Neg(n)`), i.e. inverted, and
   generatedContacts measures the other piece's face behind that plane: negative pen = gap + piece thickness.
5. addManifoldContactsToContactBuffer outputs every point with pen <= contactOffset (no lower bound).
