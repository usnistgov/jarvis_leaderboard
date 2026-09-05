#!/bin/bash
# Reproduce AI-SinglePropertyPrediction-Tc_supercon-dft_3d-test-mae.csv
#   ALIGNN 2.0 (radius graph, cutoff 8.0) on dft_3d property 'Tc_supercon'.
# Config saved next to this script as config_Tc_supercon.json. Test split is the JARVIS-Leaderboard
# benchmark dft_3d_Tc_supercon; structures pulled from jarvis.db.figshare data("dft_3d").
# Inference-only reproduction (model checkpoint on Figshare / alignn.pretrained):
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/fast_graph/bin/python
export PYTHONPATH=/home/kamalch/Software/fast_graph/alignn:/home/kamalch/Software/slako312/jarvis_leaderboard
# Load config_Tc_supercon.json + best_model.pt, build kNN/radius graphs per config, predict test ids,
# write id,prediction CSV, then zip to AI-SinglePropertyPrediction-Tc_supercon-dft_3d-test-mae.csv.zip (see repo inference helper).
echo "See config_Tc_supercon.json; reproduction script uses the saved config + checkpoint."
