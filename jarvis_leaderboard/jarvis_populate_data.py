#!/usr/bin/env python

"""Module to train for a folder with formatted dataset."""
import csv
import os
import sys
import time
import json
import zipfile
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson, dumpjson
import argparse
from jarvis.db.figshare import data
import jarvis_leaderboard

root_dir = str(
    jarvis_leaderboard.__path__[0]
)  # os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="JARVIS-Leaderboard")
parser.add_argument(
    "--benchmark_file",
    default="AI-SinglePropertyPrediction-exfoliation_energy-dft_3d-test-mae",
    # default="SinglePropertyPrediction-test-exfoliation_energy-dft_3d-AI-mae",
    help="Benchmarks available in jarvis_leaderboard/benchmarks/*/*.zip",
)
parser.add_argument(
    "--id_tag",
    default="jid",
    help="Name of identifier in a dataset: id or jid",
)
parser.add_argument(
    "--output_path",
    default="DataPath",
    help="Path for storing the training data.",
)


def get_val(df=None, id_tag="jid", prop="", jv_id="JVASP-14441"):
    """Get data from dataframe."""
    return df[df[id_tag] == jv_id][prop].values[0]


def get_dataset(
    benchmark_file="",
    dataset="",
    output_path="",
    prop="",
    method="",
    task="",
    id_tag="",
    filename="dataset_info.json",
):
    b_info = {}
    b_info["benchmark_file"] = benchmark_file
    b_info["dataset"] = dataset
    b_info["output_path"] = output_path
    b_info["prop"] = prop
    b_info["methods"] = method
    b_info["id_tag"] = id_tag

    temp = dataset + "_" + prop + ".json.zip"
    temp2 = dataset + "_" + prop + ".json"
    fname = os.path.join(root_dir, "benchmarks", method, task, temp)
    # fname = os.path.join(root_dir, "dataset", method, task, temp)
    # fname = os.path.join("jarvis_leaderboard", "dataset", method, task, temp)
    print("dataset file to be used", fname)
    if dataset in ["dft_3d", "dft_2d", "qe_tb"]:
        dat = data(dataset)
        info = {}
        for i in dat:
            info[i[id_tag]] = Atoms.from_dict(i["atoms"])

        zp = zipfile.ZipFile(fname)
        train_val_test = json.loads(zp.read(temp2))
        # print(train_val_test)
        train = train_val_test["train"]
        val = {}
        if "val" in train_val_test:
            val = train_val_test["val"]
        test = train_val_test["test"]
        cwd = os.getcwd()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # os.chdir(output_path)
        id_prop = os.path.join(output_path, "id_prop.csv")
        f = open(id_prop, "w")
        for i, j in train.items():
            line = str(i) + "," + str(j) + "\n"
            f.write(line)
            pos_name = os.path.join(output_path, str(i))
            info[i].write_poscar(pos_name)
        for i, j in val.items():
            line = str(i) + "," + str(j) + "\n"
            f.write(line)
            pos_name = os.path.join(output_path, str(i))
            info[i].write_poscar(pos_name)

        for i, j in test.items():
            line = str(i) + "," + str(j) + "\n"
            f.write(line)
            pos_name = os.path.join(output_path, str(i))
            info[i].write_poscar(pos_name)
        f.close()
        print("number of training samples", len(train))
        print("number of validation samples", len(val))
        print("number of test samples", len(test))
        b_info["n_train"] = len(train)
        b_info["n_val"] = len(val)
        b_info["n_test"] = len(test)
        filename = os.path.join(output_path, filename)
        dumpjson(data=b_info, filename=filename)
        return info
        # os.chdir(cwd)


# jarvis_leaderboard/dataset/AI/PP/dft_3d_exfoliation_energy.json.zip
if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    benchmark_file = args.benchmark_file
    method = benchmark_file.split("-")[0]
    task = benchmark_file.split("-")[1]
    prop = benchmark_file.split("-")[2]
    dataset = benchmark_file.split("-")[3]
    output_path = args.output_path
    # method = benchmark_file.split("-")[4]
    # task = benchmark_file.split("-")[0]
    id_tag = args.id_tag
    print("benchmark_file", benchmark_file)
    print("dataset", dataset)
    print("output_path", output_path)
    print("property", prop)
    print("method", method)
    print("task", task)
    print("id_tag", id_tag)

    info = get_dataset(
        benchmark_file=benchmark_file,
        dataset=dataset,
        output_path=output_path,
        prop=prop,
        method=method,
        task=task,
        id_tag=id_tag,
    )
