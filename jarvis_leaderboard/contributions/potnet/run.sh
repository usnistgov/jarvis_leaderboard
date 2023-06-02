# Running code for evaluation of formation energy peratom on JARVIS leaderboard
jarvis_populate_data.py --benchmark_file AI-SinglePropertyPrediction-formation_energy_peratom-dft_3d --output_path=Out
python main.py --config configs/potnet.yaml --output_dir output --data_root Out/ --checkpoint checkpoints/formation_energy_peratom/checkpoint.pt --testing
