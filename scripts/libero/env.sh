# SimuGuard x LIBERO runtime on H800-B.  Usage: source env.sh [official|rpent]
# official: Cap-bench's official LIBERO stack (py3.10, robosuite 1.4.0, mujoco 3.2.3, LIBERO 8f1084e)
# rpent:    rpent-venv (py3.11, robosuite 1.5.2 with lite_physics, mujoco 3.3.0, rpent_libero 0.2.0)
# Both render with Mesa software EGL (no GPU).
STACK=${1:-official}
if [ "$STACK" = official ]; then
  source /path/to/libero-runtime/libero_runtime.cpu.env
else
  source /path/to/libero-runtime/libero_runtime.rpentvenv.cpu.env
fi
export PY=$CAPB_LIBERO_PY
export SG=/path/to/simuguard-libero
export RUNS=/path/to/simuguard-libero-runs
export FFMPEG=/path/to/ffmpeg-static/ffmpeg
export LP_NUM_THREADS=${LP_NUM_THREADS:-4}
