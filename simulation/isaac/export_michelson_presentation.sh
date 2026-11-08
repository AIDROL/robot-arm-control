#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
isaac_python=${ISAAC_SIM_PYTHON:-"$HOME/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh"}
if [[ ! -x "$isaac_python" ]]; then
  printf 'Isaac Sim python.sh not found: %s\nSet ISAAC_SIM_PYTHON to your Isaac Sim python.sh path.\n' "$isaac_python" >&2
  exit 1
fi
mkdir -p runs
printf 'Exporting synchronized replay. Log: %s/runs/isaac-presentation-export.log\n' "$PWD"
exec >runs/isaac-presentation-export.log 2>&1
exec env -u PYTHONPATH -u ROS_DISTRO -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  "$isaac_python" \
  simulation/isaac/export_michelson_presentation.py "$@"
