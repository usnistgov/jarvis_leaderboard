#!/bin/bash
# QV-JLB-Blend-v1: composition-only predictions for two SinglePropertyPrediction benchmarks.
# Installs the pinned versions, then writes the two csv.zip files next to this script.
set -e
cd "$(dirname "$0")"
pip install -r requirements.txt
python run.py mxene275_formation_energy
python run.py halide_peroskites_HSE_gap
