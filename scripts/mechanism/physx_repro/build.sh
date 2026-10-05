#!/bin/bash
# Builds PhysX 5.3.1 foundation + common (geomutils) + cooking with g++ (no cmake/packman), then repro.cpp.
D=/mnt/nvme0/shared/USER/simuguard/probe/physx_repro; P=$D/PhysX-105.1-physx-5.3.1/physx; B=$D/build; mkdir -p $B/obj
INC="-I$P/include -I$P/source/foundation/include -I$P/source/common/include -I$P/source/common/src -I$P/source/physxgpu/include -I$P/source/physxcooking/src"
for d in $(find $P/source/geomutils/include $P/source/geomutils/src -type d); do INC="$INC -I$d"; done
FLAGS="-std=c++11 -O3 -fno-rtti -fno-exceptions -fno-strict-aliasing -fPIC -w -DNDEBUG -DPX_SUPPORT_PVD=0 -DPX_SUPPORT_OMNI_PVD=0 -DPX_PHYSX_STATIC_LIB -DPX_PUBLIC_RELEASE=1 -DDISABLE_CUDA_PHYSX -DPX_COOKING ${EXTRA_FLAGS}"
echo "$INC" > $B/inc.txt; echo "$FLAGS" > $B/flags.txt
SRC=$( (ls $P/source/foundation/*.cpp $P/source/foundation/unix/*.cpp $P/source/common/src/*.cpp $P/source/physxcooking/src/*.cpp; find $P/source/geomutils/src -name '*.cpp') | grep -v -i windows)
if [ -n "$1" ]; then SRC=$(echo "$SRC" | grep -E "$1"); fi
echo "$SRC" | xargs -P 48 -I{} bash -c 'o='$B'/obj/$(echo {} | md5sum | cut -c1-8)_$(basename {} .cpp).o; g++ '"$FLAGS $INC"' -c {} -o $o 2>>'$B'/err.log || echo FAIL {}'
rm -f $B/libphysx_min.a; ar rcs $B/libphysx_min.a $B/obj/*.o
g++ $FLAGS $INC $D/repro.cpp -o $B/repro $B/libphysx_min.a -lpthread -ldl 2>&1 | grep -E "error|undefined" | head -30
ls -la $B/repro $B/libphysx_min.a 2>&1 | cut -c1-120
