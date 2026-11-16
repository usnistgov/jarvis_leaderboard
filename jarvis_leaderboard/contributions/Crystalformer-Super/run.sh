# Running code example for evaluation of formation energy peratom on JARVIS leaderboard
# (JARVIS dft_3d_2021, standard 44578/5572/5572 split; 800 epochs, global batch 256, 2 GPUs, seed 123)
git clone https://github.com/Rachellincn/Crystalformer-Super
cd Crystalformer-Super
pip install -r requirements.txt
export CUDA_PATH=/usr/local/cuda-12.1
GPUS=0,1 CRYSTALFORMER_DATA=$PWD/jarvis_data RUN_DIR=runs/seed123 bash scripts/run_full.sh
# the prediction file is written to runs/seed123/benchmark/test/AI-SinglePropertyPrediction-formation_energy_peratom-dft_3d-test-mae.csv.zip
