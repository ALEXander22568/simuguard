#!/bin/bash
V=/mnt/nvme0/shared/USER/simuguard/probe/physx_repro_versions; cd $V
base=105.1-physx-5.3.1; new=106.0-physx-5.4.0
for name in simplex_eps both; do
  rm -rf var_$name; mkdir -p var_$name; cp -r $base/PhysX-$base var_$name/PhysX-var_$name; cp build_tag.sh var_$name/
  S=var_$name/PhysX-var_$name/physx/source/geomutils/src
  cp $new/PhysX-$new/physx/source/geomutils/src/gjk/GuGJKSimplex.h $S/gjk/GuGJKSimplex.h
  [ $name = both ] && sed -i 's|const Vec3V extent1 = V3Mul(V3LoadU(hullData0->mInternal.mExtents), vScale1);|const Vec3V extent1 = V3Mul(V3LoadU(hullData1->mInternal.mExtents), vScale1);|' $S/pcm/GuPCMContactConvexConvex.cpp
  echo "--- variant $name: files differing from clean 5.3.1: $(diff -rq $base/PhysX-$base var_$name/PhysX-var_$name | wc -l); $(diff $base/PhysX-$base/physx/source/geomutils/src/gjk/GuGJKSimplex.h $S/gjk/GuGJKSimplex.h | grep -c '^[<>]') lines in GuGJKSimplex.h"
  bash build_tag.sh var_$name > var_$name/build.log 2>&1; echo "    FAIL lines: $(grep -c FAIL var_$name/build.log); binary: $(ls var_$name/build/repro 2>/dev/null | wc -l)"
  for e in fig1 ep3 ep5 ep14; do o=$(python3 -c "import json;print(json.load(open('../physx_repro/data/$e/info.json'))['onset'])"); echo "   $e onset $o: [$(var_$name/build/repro ../physx_repro/data/$e 2>&1 | grep "step $o:" | cut -c1-120)] calls: $(var_$name/build/repro ../physx_repro/data/$e 2>&1 | grep -c 'ret 1')"; done
done
echo "--- 5.4.0 binary present: $(ls $new/build/repro | wc -l); fig1 contact lines on 5.4.0: $($new/build/repro ../physx_repro/data/fig1 2>&1 | grep -c 'ret 1'), on clean 5.3.1: $($base/build/repro ../physx_repro/data/fig1 2>&1 | grep -c 'ret 1')"
$new/build/repro ../physx_repro/data/ep3 2>&1 | grep -E 'step 126(1[4-9]|2)' | cut -c1-140
