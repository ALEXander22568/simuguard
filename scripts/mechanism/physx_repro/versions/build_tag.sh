#!/bin/bash
# Same minimal build as physx_repro/build.sh, against the newest PhysX tag (110.1-omni-and-physx-5.9.0), unpatched sources.
set -u
D=/mnt/nvme0/shared/USER/simuguard/probe/physx_repro_versions/$1; mkdir -p $D && cd $D
T=$1
[ -d PhysX-$T ] || { curl -sL -m 600 -o physx.tar.gz https://github.com/NVIDIA-Omniverse/PhysX/archive/refs/tags/$T.tar.gz && tar xzf physx.tar.gz PhysX-$T/physx/include PhysX-$T/physx/source/foundation PhysX-$T/physx/source/common PhysX-$T/physx/source/geomutils PhysX-$T/physx/source/physxcooking PhysX-$T/physx/source/physxgpu/include PhysX-$T/physx/version.txt 2>/dev/null; rm -f physx.tar.gz; }
cp ../../physx_repro/repro.cpp repro.cpp
P=$D/PhysX-$T/physx; B=$D/build; mkdir -p $B/obj
INC="-I$P/include -I$P/source/foundation/include -I$P/source/common/include -I$P/source/common/src -I$P/source/physxgpu/include -I$P/source/physxcooking/src"
for d in $(find $P/source/geomutils/include $P/source/geomutils/src -type d); do INC="$INC -I$d"; done
FLAGS="-std=c++14 -O3 -fno-rtti -fno-exceptions -fno-strict-aliasing -fPIC -w -DNDEBUG -DPX_SUPPORT_PVD=0 -DPX_SUPPORT_OMNI_PVD=0 -DPX_PHYSX_STATIC_LIB -DPX_PUBLIC_RELEASE=1 -DDISABLE_CUDA_PHYSX -DPX_COOKING"
SRC=$( (ls $P/source/foundation/*.cpp $P/source/foundation/unix/*.cpp $P/source/common/src/*.cpp $P/source/physxcooking/src/*.cpp; find $P/source/geomutils/src -name '*.cpp') 2>/dev/null | grep -v -i windows)
echo "sources: $(echo "$SRC" | wc -l)"; rm -f $B/err.log
echo "$SRC" | xargs -P 32 -I{} bash -c 'o='$B'/obj/$(echo {} | md5sum | cut -c1-8)_$(basename {} .cpp).o; g++ '"$FLAGS $INC"' -c {} -o $o 2>>'$B'/err.log || echo FAIL {}' | head -20
rm -f $B/libphysx_min.a; ar rcs $B/libphysx_min.a $B/obj/*.o
g++ $FLAGS $INC $D/repro.cpp -o $B/repro $B/libphysx_min.a -lpthread -ldl 2>&1 | grep -E "error|undefined" | head -30
ls -la $B/repro 2>&1 | cut -c1-120; cat $P/version.txt 2>/dev/null
for e in fig1 ep3 ep5 ep14; do echo "== $e (PhysX $T)"; $B/repro ../../physx_repro/data/$e 2>&1 | grep -E "sep -0\.0[1-9]" | head -3 | cut -c1-170; done
