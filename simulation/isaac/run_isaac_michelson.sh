#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
isaac_python=${ISAAC_SIM_PYTHON:-"$HOME/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh"}
if [[ ! -x "$isaac_python" ]]; then
  printf 'Isaac Sim python.sh not found: %s\nSet ISAAC_SIM_PYTHON to your Isaac Sim python.sh path.\n' "$isaac_python" >&2
  exit 1
fi
mkdir -p runs
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ ${args[i]} == --output ]]; then
    output=${args[i+1]-}
  elif [[ ${args[i]} == --output=* ]]; then
    output=${args[i]#--output=}
  fi
done
if [[ -n ${output-} && ( -e $output || -L $output ) ]]; then
  printf 'Isaac output directory already exists: %s\nChoose a new --output path; the existing run is preserved.\n' "$output" >&2
  exit 2
fi
printf 'Starting Isaac Sim. Run log: %s/runs/isaac-michelson-current.log\n' "$PWD"
exec >runs/isaac-michelson-current.log 2>&1
exec env -u PYTHONPATH -u ROS_DISTRO -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  "$isaac_python" \
  simulation/isaac/isaac_michelson.py "$@"
