#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
movie=${1:-runs/isaac-michelson-presentation-20261104/michelson-complete.mp4}
if [[ ! -f "$movie" ]]; then
  printf 'Presentation movie not found: %s\nExport it with export_michelson_presentation.py first.\n' "$movie" >&2
  exit 2
fi
exec ./.venv/bin/python simulation/isaac/play_michelson_presentation.py --movie "$movie"
