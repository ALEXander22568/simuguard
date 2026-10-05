#!/bin/bash
# 5.3.1 with a print in the two triangle tests of the GJK simplex solver: the value compared with FEps at the onset step
V=/mnt/nvme0/shared/USER/simuguard/probe/physx_repro_versions; cd $V
base=105.1-physx-5.3.1; rm -rf var_trace; mkdir -p var_trace; cp -r $base/PhysX-$base var_trace/PhysX-var_trace; cp build_tag.sh var_trace/
python3 - var_trace/PhysX-var_trace/physx/source/geomutils/src/gjk/GuGJKSimplex.h <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
old="\t\tif(FAllGrtrOrEq(eps, area))\n"
new='\t\t{ extern int sg_trace; if(sg_trace) { float _a; FStore(area, &_a); float _ab, _ac; FStore(V3Length(ab), &_ab); FStore(V3Length(ac), &_ac); printf("[simplex] triangle |ab x ac|^2 = %.3e (FEps = 1.192e-07, 5.4.0 threshold 1.421e-14) |ab| = %.2f mm |ac| = %.2f mm -> %s\\n", _a, _ab*1e3f, _ac*1e3f, _a <= 1.1920929e-7f ? "DEGENERATE in 5.3.1" : "ok"); } }\n'+old
assert s.count(old)==2; s=s.replace(old,new); s=s.replace("namespace physx\n{","#include <cstdio>\nnamespace physx\n{",1); open(p,"w").write(s)
PY
python3 - var_trace/repro.cpp.tmp <<'PY'
PY
bash build_tag.sh var_trace > var_trace/build.log 2>&1
# repro.cpp is copied by build_tag.sh; add the trace switch and rebuild only the program
python3 - var_trace/repro.cpp <<'PY'
import sys,re
p=sys.argv[1]; s=open(p).read()
if "int sg_trace" not in s:
    s=s.replace("#include <cstdio>","#include <cstdio>\nint sg_trace = 0;",1)
    # switch on for the step given in SG_TRACE_STEP
    s=re.sub(r"(if \(getenv\(\"SG_CLEAR_WARM_AT\"\))", r'sg_trace = (getenv("SG_TRACE_STEP") && atoi(getenv("SG_TRACE_STEP")) == step) ? 1 : 0;\n    \1', s, count=1)
    open(p,"w").write(s)
print("patched repro:", "sg_trace = (getenv" in s)
PY
B=var_trace/build; P=var_trace/PhysX-var_trace/physx
INC="-I$P/include -I$P/source/foundation/include -I$P/source/common/include -I$P/source/common/src -I$P/source/physxgpu/include -I$P/source/physxcooking/src"; for d in $(find $P/source/geomutils/include $P/source/geomutils/src -type d); do INC="$INC -I$d"; done
FLAGS="-std=c++14 -O3 -fno-rtti -fno-exceptions -fno-strict-aliasing -fPIC -w -DNDEBUG -DPX_SUPPORT_PVD=0 -DPX_SUPPORT_OMNI_PVD=0 -DPX_PHYSX_STATIC_LIB -DPX_PUBLIC_RELEASE=1 -DDISABLE_CUDA_PHYSX -DPX_COOKING"
g++ $FLAGS $INC var_trace/repro.cpp -o $B/repro $B/libphysx_min.a -lpthread -ldl 2>&1 | grep -E "error|undefined" | head; grep -c FAIL var_trace/build.log
for e in fig1 ep3 ep5 ep14; do o=$(python3 -c "import json;print(json.load(open('../physx_repro/data/$e/info.json'))['onset'])"); echo "== $e onset $o"; SG_TRACE_STEP=$o $B/repro ../physx_repro/data/$e 2>&1 | grep -E "simplex|step $o:" | cut -c1-200 | head -12; done
