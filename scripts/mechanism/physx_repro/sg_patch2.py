"""Second set of prints: GJK loop (GuGJKPenetration.h), signed projections in the witness face choice, and repro options."""
import shutil
from pathlib import Path
D = Path("/mnt/nvme0/shared/USER/simuguard/probe/physx_repro")
P = D / "PhysX-105.1-physx-5.3.1/physx/source/geomutils/src"
def edit(f, edits, from_orig=True):
    o = f.with_suffix(f.suffix + ".orig")
    if from_orig and not o.exists(): shutil.copy(f, o)
    s = (o if from_orig else f).read_text()
    for a, b in edits:
        assert s.count(a) == 1, (f.name, s.count(a), a[:80]); s = s.replace(a, b)
    f.write_text(s); print("patched", f.name)
F = lambda x: "([&]{physx::PxReal t_; physx::aos::FStore(%s,&t_); return t_;}())" % x
edit(P / "gjk/GuGJKPenetration.h", [
 ('#ifndef GU_GJK_PENETRATION_H\n#define GU_GJK_PENETRATION_H\n', '#ifndef GU_GJK_PENETRATION_H\n#define GU_GJK_PENETRATION_H\n#include <cstdio>\nextern int gSgVerbose;\n'),
 ('''			bNotTerminated = FIsGrtr(dist, eps);
''', '''			bNotTerminated = FIsGrtr(dist, eps);
			if(gSgVerbose) printf("[gjk] warm start: %%d points in, simplex size after solve %%d, dist=%%.6f eps=%%.7f contactDist+margins=%%.5f\\n", int(warmStartSize), int(size), %s, %s, %s);
''' % (F("dist"), F("eps"), F("sumExpandedMargin"))),
 ('''			const FloatV vw = V3Dot(v, support);
''', '''			const FloatV vw = V3Dot(v, support);
			if(gSgVerbose) printf("[gjk] iter: simplex size %%d dist=%%.6f vw=%%.6f (NON_INTERSECT if vw > %%.5f; converged if vw > %%.6f) new support idx a=%%d b=%%d\\n", int(size), %s, %s, %s, %s, int(aInd[size]), int(bInd[size]));
''' % (F("dist"), F("vw"), F("sumExpandedMargin"), F("FMul(dist, relDif)"))),
 ('''			bNotDegenerated = FIsGrtr(prevDist, dist);
''', '''			bNotDegenerated = FIsGrtr(prevDist, dist);
			if(gSgVerbose) printf("[gjk] after simplex solve: size %%d dist=%%.6f prevDist=%%.6f -> %%s\\n", int(size), %s, %s, BAllEqTTTT(bNotDegenerated) ? "progress" : "NO PROGRESS (degenerate exit, no test against contactDist)");
''' % (F("dist"), F("prevDist"))),
])
# signed projections (applied on top of the first patch)
edit(P / "pcm/GuPCMContactGenBoxConvex.cpp", [
 ('''		if(gSgVerbose){ printf("[full/gjk-normal] faceIndex1''', '''		if(gSgVerbose){ PxVec3 cA; V3StoreU(M33MulV3(map0->shape2Vertex, transform0To1V.transformInv(closestA)), cA); PxVec3 cB; V3StoreU(M33MulV3(map1->shape2Vertex, closestB), cB);
			printf("[full/gjk-normal] signed dot(A face normal, gjk normal in A)=%.4f (facing B would be negative); signed dot(B face normal, gjk normal)=%.4f (facing A is positive)\\n", sgF(V3Dot(incidentNormal, normalIn0)), sgF(V3Dot(referenceNormal, normal)));
			printf("[full/gjk-normal] distance of witness point to each face plane of A (vertex space, m):"); for(PxU32 i=0;i<polyData0.mNbPolygons;++i) printf(" f%d=%.5f", int(i), polyData0.mPolygons[i].mPlane.distance(cA)); printf("  toleranceA=%.5f\\n", toleranceA);
			printf("[full/gjk-normal] distance of witness point to each face plane of B:"); for(PxU32 i=0;i<polyData1.mNbPolygons;++i){ PxReal dd = polyData1.mPolygons[i].mPlane.distance(cB); if(dd > -0.002f) printf(" f%d=%.5f", int(i), dd);} printf("  toleranceB=%.5f\\n", toleranceB); }
		if(gSgVerbose){ printf("[full/gjk-normal] faceIndex1'''),
], from_orig=False)
edit(D / "repro.cpp", [
 ('''    if (gSgVerbose) printf("--- step %d\\n", step);
''', '''    if (gSgVerbose) printf("--- step %d\\n", step);
    if (getenv("SG_CLEAR_WARM_AT") && atoi(getenv("SG_CLEAR_WARM_AT")) == step) { cache.getManifold().mNumWarmStartPoints = 0; printf("warm start cleared before step %d\\n", step); }
    if (getenv("SG_CLEAR_WARM_ALWAYS")) cache.getManifold().mNumWarmStartPoints = 0;
'''),
])
