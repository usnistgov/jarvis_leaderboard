#!/bin/bash
# QV-JLB-Site-Blend-v1: adsorption-site predictions for the SinglePropertyPrediction benchmark tinnet_O_ead.
# Installs the pinned versions, then writes the csv.zip file next to this script.
set -e
cd "$(dirname "$0")"
pip install -r requirements.txt
python run.py tinnet_O_ead
