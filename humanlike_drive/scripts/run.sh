#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
scene_python="${HUMANLIKE_PYTHON:-/home/nubot/anaconda3/envs/mujoco/bin/python}"
export PYTHONPATH="${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"
cd -- "${project_dir}"
exec "${scene_python}" "$@"
