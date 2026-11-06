#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
carla_dir="${CARLA_ROOT:-$project_dir/third_party/vendor/carla-0.9.15}"
python_bin="${DRIVE_PYTHON:-/home/nubot/anaconda3/envs/mujoco/bin/python}"
api_dir="$project_dir/third_party/vendor/carla-python"
if [[ ! -d "$api_dir/carla" ]]; then
  wheel_path=""
  for wheel_dir in "$carla_dir/PythonAPI/carla/dist" "$project_dir/third_party/vendor/wheels"; do
    if [[ -d "$wheel_dir" ]]; then
      wheel_path="$(find "$wheel_dir" -maxdepth 1 -name '*cp310*linux*.whl' -print -quit)"
      [[ -n "$wheel_path" ]] && break
    fi
  done
  if [[ -z "$wheel_path" ]]; then
    echo "Matching Python 3.10 CARLA wheel missing; download carla==0.9.15 to third_party/vendor/wheels" >&2
    exit 1
  fi
  "$python_bin" -m pip install --no-deps --target "$api_dir" "$wheel_path"
fi
export PYTHONPATH="$api_dir:$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" "$@"
