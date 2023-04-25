conda activate /wrk/knc6/Software/kgcnn
jarvis_populate_data.py --benchmark_file SinglePropertyPrediction-test-exfoliation_energy-dft_3d-AI-mae --output_path=Out
python run.py
