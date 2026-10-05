#!/bin/bash
# NOTE: only the `extent1` variant of this script is valid (5.3.1 with the hullData1 extent fix alone: all four
# onsets still reproduce).  Its `simplex_eps` and `both` variants replace one `eps` too many and do not compile;
# variant2.sh, which copies the 5.4.0 header instead, supersedes them.
# variants of the clean 5.3.1 tree with single changes taken from 5.4.0, to see which one removes the onset contact
V=/mnt/nvme0/shared/USER/simuguard/probe/physx_repro_versions; cd $V
base=105.1-physx-5.3.1
grep -n "FEps()" -B6 -A12 $base/PhysX-$base/physx/source/geomutils/src/gjk/GuGJKSimplex.h | sed -n 1,60p
for name in simplex_eps extent1 both; do
  rm -rf var_$name; mkdir -p var_$name; cp -r $base/PhysX-$base var_$name/PhysX-var_$name; cp build_tag.sh var_$name/
  S=var_$name/PhysX-var_$name/physx/source/geomutils/src
  if [ $name != extent1 ]; then
    python3 - $S/gjk/GuGJKSimplex.h <<'PY'
import sys,re
p=sys.argv[1]; s=open(p).read()
n1=s.count("const FloatV eps = FEps();"); n2=s.count("if(FAllGrtrOrEq(eps, area))")
s=s.replace("const FloatV eps = FEps();","const FloatV eps2 = FLoad(PX_EPS_REAL * PX_EPS_REAL);").replace("if(FAllGrtrOrEq(eps, area))","if(FAllGrtrOrEq(eps2, area))")
open(p,"w").write(s); print("simplex eps replaced:", n1, n2)
PY
  fi
  if [ $name != simplex_eps ]; then
    sed -i 's|const Vec3V extent1 = V3Mul(V3LoadU(hullData0->mInternal.mExtents), vScale1);|const Vec3V extent1 = V3Mul(V3LoadU(hullData1->mInternal.mExtents), vScale1);|' $S/pcm/GuPCMContactConvexConvex.cpp
    grep -c "hullData1->mInternal.mExtents), vScale1" $S/pcm/GuPCMContactConvexConvex.cpp
  fi
  bash build_tag.sh var_$name 2>&1 | grep -E 'FAIL|error|undefined' | head -5
  echo "--- variant $name:"
  for e in fig1 ep3 ep5 ep14; do o=$(python3 -c "import json;print(json.load(open('../physx_repro/data/$e/info.json'))['onset'])"); echo "   $e onset $o: $(var_$name/build/repro ../physx_repro/data/$e 2>&1 | grep "step $o:" | cut -c1-130)"; done
done
