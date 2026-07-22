#!/bin/bash
# ----------------------------------------------------------------------
# Reproduce the ClosureGap contribution for the JARVIS-Leaderboard
# benchmark: AI-SinglePropertyPrediction-optb88vdw_bandgap-dft_3d-test-mae
#
# It trains ClosureGap on the official train split and writes the official
# id,target,prediction CSV (zipped). ClosureGap itself is stdlib-only; the
# only heavy dependency is jarvis-tools, used to map JVASP ids to formulae.
# ----------------------------------------------------------------------
set -euo pipefail

# 1. Install dependencies (ClosureGap is vendored alongside this script).
pip install jarvis-tools

# 2. Decode the official benchmark split shipped in the leaderboard repo.
#    (json.zip -> json). Adjust the path if running outside the repo tree.
BENCH_ZIP="${BENCH_ZIP:-dft_3d_optb88vdw_bandgap.json.zip}"
python -c "import zipfile,sys; zipfile.ZipFile('${BENCH_ZIP}').extractall('.')"

# 3. Run the adapter: train, predict, and package the submission csv.zip.
python run.py \
  --bench-json dft_3d_optb88vdw_bandgap.json \
  --cache-dir ./data \
  --out-dir ./output \
  --epochs 900

echo "Done. Submission file is in ./output/"
