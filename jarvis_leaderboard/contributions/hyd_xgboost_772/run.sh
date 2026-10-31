#!/usr/bin/env bash
set -euo pipefail
: "${JARVIS_RAW:?Set JARVIS_RAW to jdft_3d-12-12-2022.json}"
: "${JARVIS_SPLIT:?Set JARVIS_SPLIT to dft_3d_magmom_oszicar.json.zip}"
here="$(cd "$(dirname "$0")" && pwd)"
python "$here/run.py" --raw "$JARVIS_RAW" --split "$JARVIS_SPLIT" --output "${JARVIS_OUTPUT:-$here}"
