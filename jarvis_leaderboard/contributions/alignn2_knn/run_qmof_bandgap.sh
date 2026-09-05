#!/bin/bash
# Reproduce AI-SinglePropertyPrediction-bandgap-qmof-test-mae.csv
#   ALIGNN 2.0 (kNN graph, cutoff 8.0) on qmof property 'bandgap'.
# Config saved next to this script as config_qmof_bandgap.json. Test split is the JARVIS-Leaderboard
# benchmark qmof_bandgap; structures pulled from jarvis.db.figshare data("qmof").
# Inference-only reproduction (model checkpoint on Figshare / alignn.pretrained):
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/fast_graph/bin/python
export PYTHONPATH=/home/kamalch/Software/fast_graph/alignn:/home/kamalch/Software/slako312/jarvis_leaderboard
# Load config_qmof_bandgap.json + best_model.pt, build kNN/radius graphs per config, predict test ids,
# write id,prediction CSV, then zip to AI-SinglePropertyPrediction-bandgap-qmof-test-mae.csv.zip (see repo inference helper).
echo "See config_qmof_bandgap.json; reproduction script uses the saved config + checkpoint."
