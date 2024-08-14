#!/bin/bash
pip install -q jarvis-tools jarvis-leaderboard
git clone https://github.com/QEF/q-e.git
cd q-e
make pw
cd ..
info = qejob_relax.runjob()