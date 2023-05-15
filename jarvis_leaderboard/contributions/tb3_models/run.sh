#pip install jarvis_leaderboard
#conda create --name /wrk/knc6/Software/tb3 python=3.8 chardet
#conda activate /wrk/knc6/Software/tb3
#conda install -c conda-forge julia=1.6.2
#pip install requests
#git clone https://github.com/usnistgov/tb3py.git
#cd tb3py
#python setup.py develop

jarvis_populate_data.py --benchmark_file  ES-SinglePropertyPrediction-bandgap_JVASP_1002_Si-dft_3d-test-mae.csv.zip
#jarvis_populate_data.py --benchmark_file  ES-SinglePropertyPrediction-bandgap-dft_3d-test-mae.csv.zip
conda activate /wrk/knc6/Software/tb3
