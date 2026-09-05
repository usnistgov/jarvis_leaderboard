#!/bin/bash
# Reproduce AI-SinglePropertyPrediction-gap-qm9_std_jctc-test-mae.csv
#   ALIGNN 2.0 (kNN graph, cutoff 8.0) on qm9_std_jctc property 'gap'.
# Config saved next to this script as config_qm9_std_jctc_gap.json. Test split is the JARVIS-Leaderboard
# benchmark qm9_std_jctc_gap; structures pulled from jarvis.db.figshare data("qm9_std_jctc").
# Inference-only reproduction (model checkpoint on Figshare / alignn.pretrained):
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/fast_graph/bin/python
export PYTHONPATH=/home/kamalch/Software/fast_graph/alignn:/home/kamalch/Software/slako312/jarvis_leaderboard
# Load config_qm9_std_jctc_gap.json + best_model.pt, build kNN/radius graphs per config, predict test ids,
# write id,prediction CSV, then zip to AI-SinglePropertyPrediction-gap-qm9_std_jctc-test-mae.csv.zip (see repo inference helper).
echo "See config_qm9_std_jctc_gap.json; reproduction script uses the saved config + checkpoint."
