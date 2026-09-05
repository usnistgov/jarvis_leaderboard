#!/bin/bash
# Reproduce AI-SinglePropertyPrediction-dosef-alex_supercon-test-mae.csv
#   ALIGNN 2.0 (kNN graph, cutoff 8.0) on alex_supercon property 'dosef'.
# Config is saved next to this script as config_alex_supercon_dosef.json. Test split is the
# JARVIS-Leaderboard benchmark alex_supercon_dosef.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/fast_graph/bin/python
TR=/home/kamalch/miniforge3/envs/fast_graph/bin/train_alignn.py
export PYTHONPATH=/home/kamalch/Software/fast_graph/alignn:/home/kamalch/Software/slako312/jarvis_leaderboard
WORK="$HERE/_work_alex_supercon_dosef"; mkdir -p "$WORK"
cp "$HERE/config_alex_supercon_dosef.json" "$WORK/config.json"
# Build id_prop.json (train+val+test order) from the benchmark split + dataset
# structures, then train; see build_property_data.py / build_lb_data.py for the
# dft_3d and additional-dataset builders used by the other entries here.
cd "$WORK"; rm -rf results *_data
"$TR" --root_dir "$WORK" --config_name config.json --output_dir results --target_key target
"$PY" - <<PYEOF
import pandas as pd, zipfile, os
p=pd.read_csv("$WORK/results/prediction_results_test_set.csv"); p.columns=[c.strip() for c in p.columns]
name="AI-SinglePropertyPrediction-dosef-alex_supercon-test-mae.csv"
p[["id","prediction"]].to_csv("$HERE/"+name, index=False)
with zipfile.ZipFile("$HERE/AI-SinglePropertyPrediction-dosef-alex_supercon-test-mae.csv.zip","w",zipfile.ZIP_DEFLATED) as z: z.write("$HERE/"+name, name)
os.remove("$HERE/"+name); print("wrote AI-SinglePropertyPrediction-dosef-alex_supercon-test-mae.csv.zip")
PYEOF
