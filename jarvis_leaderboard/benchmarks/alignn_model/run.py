import pandas as pd
import os
from jarvis.core.atoms import Atoms
import numpy as np
from sklearn.metrics import mean_absolute_error
from jarvis.db.jsonutils import loadjson, dumpjson
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tasks = []
#for i in glob.glob("../../*/AI-SinglePropertyPrediction*test-mae.csv.zip"):
for i in glob.glob("../*/AI-SinglePropertyPrediction*test-mae.csv.zip"):
    if "formula" not in i:
        task = i.split("/")[-1].split(".csv.zip")[0]
        if task not in tasks:
            tasks.append(task)
print("tasks", tasks, len(tasks))
# For a quick test running on one task only
tasks = ["AI-SinglePropertyPrediction-exfoliation_energy-dft_3d-test-mae"]
for task in tasks:
    task = "AI-SinglePropertyPrediction-exfoliation_energy-dft_3d-test-mae"
    cmd = (
        "jarvis_populate_data.py --benchmark_file "
        + task
        + " --output_path=Out"
    )
    if not os.path.exists("Out"):
        os.system(cmd)
    dataset_info = loadjson("Out/dataset_info.json")
    n_train = dataset_info["n_train"]
    n_val = dataset_info["n_val"]
    n_test = dataset_info["n_test"]

    cmd='wget https://raw.githubusercontent.com/usnistgov/alignn/main/alignn/examples/sample_data/config_example.json'
    os.system(cmd)
    config = loadjson('config_example.json')
    config['n_train'] = n_train
    config['n_val'] = n_val
    config['n_test'] = n_test
    config['epochs'] = 20
    config['batch_size'] = 32
    dumpjson(data=config,filename="config_example.json")

    print("n_train", n_train)
    print("n_val", n_val)
    print("n_test", n_test)
    cmd='train_folder.py --root_dir "Out" --config "config_example.json" --output_dir="temp"'
    os.system(cmd)

    csv_name = task+'.csv'

    cmd='cp temp/prediction_results_test_set.csv '+csv_name
    os.system(cmd)
   
    cmd='rm -r temp'
    os.system(cmd)

    cmd = "zip " + csv_name + ".zip " + csv_name
    os.system(cmd)

    cmd = "rm config_example.json"
    os.system(cmd)
    cmd = "rm -r Out"
    os.system(cmd)
    cmd = "rm " + csv_name
    os.system(cmd)
    cmd = "rm -r " + train_dir
    os.system(cmd)
