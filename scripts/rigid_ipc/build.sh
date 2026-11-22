#!/usr/bin/env bash
# Build the Rigid IPC reference implementation (headless, no GUI, no tests). cmake comes from pip into ./tools.
set -x
cd /path/to/workspace/rigid_ipc
python3 -m pip install -q --target tools cmake==3.27.9 2>&1 | tail -2
export PATH=$PWD/tools/bin:$PATH PYTHONPATH=$PWD/tools
cmake --version
mkdir -p src/build && cd src/build
cmake -DCMAKE_BUILD_TYPE=Release -DRIGID_IPC_WITH_OPENGL=OFF -DRIGID_IPC_WITH_UNIT_TESTS=OFF -DRIGID_IPC_WITH_TOOLS=OFF .. && make -j32
echo BUILD_RC=$?
ls -la | head -30
