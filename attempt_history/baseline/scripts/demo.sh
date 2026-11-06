#!/usr/bin/env bash
# Start the installed local CARLA if needed and run the trained RGB/robot loop.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/humanlike-drive-mpl}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
for checkpoint in runs/vision/best.pt runs/ppo_track/best/best_model.zip; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing trained checkpoint: $checkpoint" >&2
    exit 1
  fi
done
server_ready() {
  bash scripts/with_carla.sh -c 'import carla; c=carla.Client("127.0.0.1",2000); c.set_timeout(2); assert c.get_server_version()=="0.9.15"' >/dev/null 2>&1
}
if ! server_ready; then
  mkdir -p outputs
  echo "Starting local CARLA; startup log: outputs/carla_server.log"
  bash scripts/start_carla.sh >> outputs/carla_server.log 2>&1 &
  server_pid=$!
  for attempt in {1..60}; do
    if server_ready; then break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "CARLA exited. See outputs/carla_server.log" >&2
      exit 1
    fi
    sleep 1
  done
  if ! server_ready; then
    echo "CARLA is not ready. See outputs/carla_server.log" >&2
    exit 1
  fi
fi
run_stamp="$(date +%Y%m%d_%H%M%S)"
exec bash scripts/with_carla.sh -u -m drive_baseline.carla_bridge \
  --mode vision-robot --vision-policy runs/vision/best.pt \
  --robot-policy runs/ppo_track/best/best_model.zip --device cuda \
  --map Town04 --steps 600 --speed 3 --output "outputs/demo_$run_stamp" "$@"
