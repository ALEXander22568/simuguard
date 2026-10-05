// Feeds one convex pair through PhysX 5.3.1's PCM convex-convex contact generation with a cache that the program
// owns and never resets (except when the inflated bounds stop overlapping, as the broadphase would).
// usage: repro DATA_DIR [order=0|1] [toleranceLength=0.1] [contactDist=0.02] [startStep=1] [verbose=0] [noBroadphaseReset=0]
#include "PxPhysicsAPI.h"
#include "cooking/PxCooking.h"
#include "GuContactMethodImpl.h"
#include "GuPersistentContactManifold.h"
#include "geomutils/PxContactBuffer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
using namespace physx;
int gSgVerbose = 0;
struct Alloc : PxAllocatorCallback {
  void* allocate(size_t size, const char*, const char*, int) override { void* p = NULL; if (posix_memalign(&p, 16, size ? size : 16)) return NULL; return p; }
  void deallocate(void* p) override { free(p); }
};
struct Err : PxErrorCallback { void reportError(PxErrorCode::Enum c, const char* m, const char* f, int l) override { fprintf(stderr, "PhysX error %d: %s (%s:%d)\n", int(c), m, f, l); } };
struct Hull { std::vector<PxVec3> v; PxVec3 scale; PxTransform local; PxConvexMesh* mesh; };
static bool loadHull(const std::string& fn, Hull& h) {
  FILE* f = fopen(fn.c_str(), "r"); if (!f) return false; int n; if (fscanf(f, "%d", &n) != 1) return false;
  h.v.resize(n); for (int i = 0; i < n; i++) fscanf(f, "%f %f %f", &h.v[i].x, &h.v[i].y, &h.v[i].z);
  char key[64]; float a[4]; h.local = PxTransform(PxIdentity);
  while (fscanf(f, "%63s", key) == 1) {
    if (!strcmp(key, "scale")) { fscanf(f, "%f %f %f", &h.scale.x, &h.scale.y, &h.scale.z); }
    else if (!strcmp(key, "local_p")) { fscanf(f, "%f %f %f", &h.local.p.x, &h.local.p.y, &h.local.p.z); }
    else if (!strcmp(key, "local_q_wxyz")) { fscanf(f, "%f %f %f %f", &a[0], &a[1], &a[2], &a[3]); h.local.q = PxQuat(a[1], a[2], a[3], a[0]); }
    else { fscanf(f, "%*[^\n]"); }
  }
  fclose(f); return true;
}
static void bounds(const Hull& h, const PxTransform& t, PxVec3& lo, PxVec3& hi) {
  lo = PxVec3(1e9f); hi = PxVec3(-1e9f);
  for (size_t i = 0; i < h.v.size(); i++) { PxVec3 p = t.transform(h.v[i].multiply(h.scale)); lo = lo.minimum(p); hi = hi.maximum(p); }
}
int main(int argc, char** argv) {
  if (argc < 2) { printf("usage\n"); return 1; }
  std::string dir = argv[1]; int order = argc > 2 ? atoi(argv[2]) : 0; float tol = argc > 3 ? atof(argv[3]) : 0.1f; float cd = argc > 4 ? atof(argv[4]) : 0.02f;
  int start = argc > 5 ? atoi(argv[5]) : 1; gSgVerbose = argc > 6 ? atoi(argv[6]) : 0; int noReset = argc > 7 ? atoi(argv[7]) : 0;
  static Alloc alloc; static Err err; PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, alloc, err); if (!fnd) { printf("no foundation\n"); return 1; }
  Hull H[2]; if (!loadHull(dir + "/hull_can.txt", H[0]) || !loadHull(dir + "/hull_basket.txt", H[1])) { printf("cannot load hulls\n"); return 1; }
  PxTolerancesScale ts; ts.length = tol; ts.speed = tol * 2.0f; PxCookingParams cp(ts);
  for (int k = 0; k < 2; k++) {
    PxConvexMeshDesc d; d.points.count = PxU32(H[k].v.size()); d.points.stride = sizeof(PxVec3); d.points.data = H[k].v.data(); d.flags = PxConvexFlag::eCOMPUTE_CONVEX;
    PxConvexMeshCookingResult::Enum res; H[k].mesh = PxCreateConvexMesh(cp, d, *PxGetStandaloneInsertionCallback(), &res);
    if (!H[k].mesh) { printf("cook failed %d\n", int(res)); return 1; }
    printf("hull %d: input verts %d, cooked verts %d, polygons %d, scale %g\n", k, int(H[k].v.size()), int(H[k].mesh->getNbVertices()), int(H[k].mesh->getNbPolygons()), H[k].scale.x);
    int same = 0; const PxVec3* cv = H[k].mesh->getVertices(); for (PxU32 i = 0; i < H[k].mesh->getNbVertices() && i < H[k].v.size(); i++) same += (cv[i] == H[k].v[i]);
    printf("   cooked vertices identical in order to input: %d\n", same);
  }
  PxConvexMeshGeometry g0(H[0].mesh, PxMeshScale(H[0].scale)), g1(H[1].mesh, PxMeshScale(H[1].scale));
  Gu::LargePersistentContactManifold* manifold = PX_PLACEMENT_NEW(alloc.allocate(sizeof(Gu::LargePersistentContactManifold), "", "", 0), Gu::LargePersistentContactManifold)();
  Gu::Cache cache; cache.setManifold(manifold); cache.getManifold().clearManifold();
  Gu::NarrowPhaseParams params(cd, 0.001f, tol);
  FILE* f = fopen((dir + "/poses.txt").c_str(), "r"); if (!f) { printf("no poses\n"); return 1; }
  int step; float a[14]; bool inRange = false; long calls = 0; int printed = 0;
  while (fscanf(f, "%d", &step) == 1) {
    for (int i = 0; i < 14; i++) fscanf(f, "%f", &a[i]);
    if (step < start) continue;
    PxTransform tc(PxVec3(a[0], a[1], a[2]), PxQuat(a[4], a[5], a[6], a[3])), tb(PxVec3(a[7], a[8], a[9]), PxQuat(a[11], a[12], a[13], a[10]));
    PxTransform wc = tc.transform(H[0].local), wb = tb.transform(H[1].local);
    PxVec3 lo0, hi0, lo1, hi1; bounds(H[0], wc, lo0, hi0); bounds(H[1], wb, lo1, hi1);
    const float infl = cd * 0.5f; bool ov = true;
    for (int k = 0; k < 3; k++) if (lo0[k] - infl > hi1[k] + infl || lo1[k] - infl > hi0[k] + infl) ov = false;
    if (!ov && !noReset) { if (inRange) { printf("step %d: bounds separated, cache cleared\n", step); cache.getManifold().clearManifold(); } inRange = false; continue; }
    if (!inRange) printf("step %d: pair in range\n", step);
    inRange = true;
    PxTransform32 t0 = order == 0 ? PxTransform32(wc) : PxTransform32(wb), t1 = order == 0 ? PxTransform32(wb) : PxTransform32(wc);
    PxContactBuffer buf; buf.count = 0; calls++;
    if (gSgVerbose) printf("--- step %d\n", step);
    if (getenv("SG_CLEAR_WARM_AT") && atoi(getenv("SG_CLEAR_WARM_AT")) == step) { cache.getManifold().mNumWarmStartPoints = 0; printf("warm start cleared before step %d\n", step); }
    if (getenv("SG_CLEAR_WARM_ALWAYS")) cache.getManifold().mNumWarmStartPoints = 0;
    bool r = Gu::pcmContactConvexConvex(order == 0 ? g0 : g1, order == 0 ? g1 : g0, t0, t1, params, cache, buf, NULL);
    float minsep = 1e9f; for (PxU32 i = 0; i < buf.count; i++) minsep = PxMin(minsep, buf.contacts[i].separation);
    if (buf.count && (minsep < -0.002f || gSgVerbose || printed < 3)) {
      printed++;
      printf("step %d: ret %d contacts %d manifoldPts %d n %.4f %.4f %.4f sep", step, int(r), int(buf.count), int(cache.getManifold().getNumContacts()), buf.contacts[0].normal.x, buf.contacts[0].normal.y, buf.contacts[0].normal.z);
      for (PxU32 i = 0; i < buf.count; i++) printf(" %.5f", buf.contacts[i].separation);
      printf("\n");
    }
  }
  printf("done: %ld narrowphase calls, last step %d\n", calls, step);
  return 0;
}
