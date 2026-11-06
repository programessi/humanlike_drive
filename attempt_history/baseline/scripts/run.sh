#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$task_dir"
export PYTHONPATH="$task_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/humanlike-drive-mpl}"
python_bin="${DRIVE_PYTHON:-/home/nubot/anaconda3/envs/mujoco/bin/python}"
exec "$python_bin" "$@"
