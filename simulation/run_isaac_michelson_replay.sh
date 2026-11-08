#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/isaac/run_isaac_michelson_replay.sh" "$@"
