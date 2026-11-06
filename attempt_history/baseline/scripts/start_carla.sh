#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
carla_dir="${CARLA_ROOT:-$project_dir/third_party/vendor/carla-0.9.15}"
cd "$carla_dir"
exec bash CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-rpc-port=2000 "$@"
