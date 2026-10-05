"""Adds debug prints (guarded by gSgVerbose) to PhysX 5.3.1 PCM convex-convex contact generation. Idempotent: restores .orig first."""
import shutil, sys
from pathlib import Path
P = Path("/mnt/nvme0/shared/USER/simuguard/probe/physx_repro/PhysX-105.1-physx-5.3.1/physx/source/geomutils/src")
HELP = '''
#include <cstdio>
extern int gSgVerbose;
namespace { 
inline void sgV(const char* tag, const physx::aos::Vec3V v) { physx::PxVec3 t; physx::aos::V3StoreU(v, t); printf(" %s=(%.5f %.5f %.5f)", tag, t.x, t.y, t.z); }
inline float sgF(const physx::aos::FloatV f) { physx::PxReal t; physx::aos::FStore(f, &t); return t; }
}
'''
def patch(rel, edits, helper_after):
    f = P / rel; o = f.with_suffix(f.suffix + ".orig")
    if not o.exists(): shutil.copy(f, o)
    s = o.read_text()
    assert helper_after in s, (rel, "helper anchor"); s = s.replace(helper_after, helper_after + HELP, 1)
    for a, b in edits:
        assert s.count(a) >= 1, (rel, a[:70]); s = s.replace(a, b)
    f.write_text(s); print("patched", rel)

# ---- GuPCMContactConvexConvex.cpp
patch("pcm/GuPCMContactConvexConvex.cpp", [
 ('''	manifold.refreshContactPoints(aToB, projectBreakingThreshold, contactDist);
''', '''	if(gSgVerbose){ printf("[cc] before refresh: pts=%d warm=%d minMargin=%.6f marginA=%.6f marginB=%.6f contactDist=%.4f", int(manifold.mNumContacts), int(manifold.mNumWarmStartPoints), sgF(minMargin), sgF(convexMargin0), sgF(convexMargin1), sgF(contactDist));
		for(PxU32 i=0;i<manifold.mNumContacts;++i) printf(" pen%d=%.5f", int(i), sgF(V4GetW(manifold.mContactPoints[i].mLocalNormalPen))); printf("\\n"); }
	manifold.refreshContactPoints(aToB, projectBreakingThreshold, contactDist);
	if(gSgVerbose){ printf("[cc] after refresh: pts=%d", int(manifold.mNumContacts));
		for(PxU32 i=0;i<manifold.mNumContacts;++i) printf(" pen%d=%.5f", int(i), sgF(V4GetW(manifold.mContactPoints[i].mLocalNormalPen))); printf("\\n"); }
'''),
 ('''	if(bLostContacts || manifold.invalidate_BoxConvex(curRTrans, transf0.q, transf1.q, minMargin, radiusA, radiusB))
	{''', '''	const PxU32 sgInval = manifold.invalidate_BoxConvex(curRTrans, transf0.q, transf1.q, minMargin, radiusA, radiusB);
	if(gSgVerbose) printf("[cc] lostContacts=%d invalidate=%d -> %s; warm=%d aIdx=%d,%d,%d,%d bIdx=%d,%d,%d,%d\\n", int(bLostContacts), int(sgInval), (bLostContacts||sgInval) ? "REGENERATE (gjk)" : "reuse cache", int(manifold.mNumWarmStartPoints),
		int(manifold.mAIndice[0]), int(manifold.mAIndice[1]), int(manifold.mAIndice[2]), int(manifold.mAIndice[3]), int(manifold.mBIndice[0]), int(manifold.mBIndice[1]), int(manifold.mBIndice[2]), int(manifold.mBIndice[3]));
	if(bLostContacts || sgInval)
	{'''),
 ('''	if(status == GJK_NON_INTERSECT)
	{
		return false;
	}''', '''	if(gSgVerbose){ printf("[cc] gjkPenetration status=%d (0 NON_INTERSECT,1 EPA_CONTACT,2 GJK_CONTACT? see GuGJKType.h) warmAfter=%d", int(status), int(manifold.mNumWarmStartPoints));
		if(status != GJK_NON_INTERSECT){ sgV("normal", output.normal); sgV("closA", output.closestA); sgV("closB", output.closestB); printf(" penDep=%.5f", sgF(output.penDep)); } printf("\\n"); }
	if(status == GJK_NON_INTERSECT)
	{
		return false;
	}'''),
 ('''		const bool fullContactGen = FAllGrtr(FLoad(0.707106781f), V3Dot(localNor, output.normal)) || (manifold.mNumContacts < initialContacts);
''', '''		const bool fullContactGen = FAllGrtr(FLoad(0.707106781f), V3Dot(localNor, output.normal)) || (manifold.mNumContacts < initialContacts);
		if(gSgVerbose){ printf("[cc] after addGJKEPAContacts: doOverlapTest=%d pts=%d initial=%d dot(localNor,normal)=%.4f fullContactGen=%d", int(doOverlapTest), int(manifold.mNumContacts), int(initialContacts), sgF(V3Dot(localNor, output.normal)), int(fullContactGen));
			sgV("normal", output.normal); printf(" penDep=%.5f\\n", sgF(output.penDep)); }
'''),
 ('''		if(numContacts > 0)
		{
			//reduce contacts
			manifold.addBatchManifoldContacts(manifoldContacts, numContacts, toleranceLength);
''', '''		if(gSgVerbose){ printf("[cc] generateFullContactManifold -> %d contacts:", int(numContacts));
			for(PxU32 i=0;i<numContacts;++i){ printf(" [pen=%.5f", sgF(V4GetW(manifoldContacts[i].mLocalNormalPen))); sgV("n", Vec3V_From_Vec4V(manifoldContacts[i].mLocalNormalPen)); printf("]"); } printf("\\n"); }
		if(numContacts > 0)
		{
			//reduce contacts
			manifold.addBatchManifoldContacts(manifoldContacts, numContacts, toleranceLength);
'''),
], 'using namespace aos;\n')

# ---- GuPCMContactGenBoxConvex.cpp
patch("pcm/GuPCMContactGenBoxConvex.cpp", [
 ('''		bool doEdgeTest = false;
			
EdgeTest:''', '''		bool doEdgeTest = false;
		if(gSgVerbose){ printf("[sat] face tests: status=%d (0 POLYDATA0,1 POLYDATA1,2 EDGE) minOverlap=%.5f feature0=%d feature1=%d", int(status), sgF(minOverlap), int(feature0), int(feature1)); sgV("minNormal", minNormal); printf("\\n"); }
			
EdgeTest:'''),
 ('''			if(status != EDGE)
				return true;
		}''', '''			if(gSgVerbose){ printf("[sat] edge test: status=%d minOverlap=%.5f", int(status), sgF(minOverlap)); sgV("minNormal", minNormal); printf("\\n"); }
			if(status != EDGE)
				return true;
		}'''),
 ('''		if(numContacts == 0 && !doEdgeTest)
		{''', '''		if(gSgVerbose) printf("[sat] contacts from status %d: %d (doEdgeTest=%d)\\n", int(status), int(numContacts), int(doEdgeTest));
		if(numContacts == 0 && !doEdgeTest)
		{'''),
 ('''		if (FAllGrtrOrEq(referenceProject, incidentProject))
		{''', '''		if(gSgVerbose){ printf("[full/gjk-normal] faceIndex1(B ref)=%d faceIndex0(A inc)=%d referenceProject=%.4f incidentProject=%.4f -> reference is %s", int(faceIndex1), int(faceIndex0), sgF(referenceProject), sgF(incidentProject), FAllGrtrOrEq(referenceProject, incidentProject) ? "B(poly1)" : "A(poly0)");
			sgV("gjkNormal", normal); sgV("refNormalB", referenceNormal); sgV("incNormalA_in0", incidentNormal); sgV("closestA", closestA); sgV("closestB", closestB); printf("\\n"); }
		if (FAllGrtrOrEq(referenceProject, incidentProject))
		{'''),
 ('''	bool doOverlapTest = false;
	if (status == GJK_DEGENERATE)
	{
		const Vec3V normal = output.normal;
		const FloatV costheta = V3Dot(output.searchDir, normal);''', '''	bool doOverlapTest = false;
	if (status == GJK_DEGENERATE)
	{
		const Vec3V normal = output.normal;
		const FloatV costheta = V3Dot(output.searchDir, normal);
		if(gSgVerbose){ const Vec3V dA = relativeConvex->getCenter(); const Vec3V dB = localConvex->getCenter(); printf("[epa] GJK_DEGENERATE: cos(searchDir,normal)=%.5f dot(centreDir,normal)=%.5f\\n", sgF(costheta), sgF(V3Dot(V3Normalize(V3Sub(dA, dB)), normal))); }'''),
 ('''		if (status == EPA_CONTACT)
		{
			addManifoldPoint(manifoldContacts, manifold, output, aToB, replaceBreakingThreshold);
		}''', '''		if(gSgVerbose){ printf("[epa] epaPenetration returned status=%d", int(status)); sgV("normal", output.normal); sgV("closA", output.closestA); sgV("closB", output.closestB); printf(" penDep=%.5f\\n", sgF(output.penDep)); }
		if (status == EPA_CONTACT)
		{
			addManifoldPoint(manifoldContacts, manifold, output, aToB, replaceBreakingThreshold);
		}'''),
], 'using namespace aos;\n')
