#conda activate /work/03943/kamalch/ls6/Software/alignn
import os,json,torch
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson, dumpjson
import json,zipfile
import zipfile
import json
import glob
import pandas as pd
import numpy as np
from jarvis.core.atoms import Atoms
import os
from alignn.ff.ff import AlignnAtomwiseCalculator, default_path, ForceField
import torch
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress
from jarvis.db.figshare import data
import subprocess
from subprocess import Popen, PIPE
mlearn = json.loads(
        zipfile.ZipFile("../mlearn.json.zip").read(
            "mlearn.json"
        )
    )
example_config = loadjson("../config_mlearn_cu.json")
example_config["n_train"] = 0 
example_config["n_val"] = 0 
example_config["n_test"] = 0
example_config["model"]["graphwise_weight"] = 1
example_config["model"]["gradwise_weight"] = 1
example_config["model"]["add_reverse_forces"] = True
example_config["model"]["lg_on_fly"] = True
example_config["model"]["alignn_layers"] = 4
example_config["epochs"] = 300
example_config["batch_size"] = 2
example_config["keep_data_order"] = True

run_dir='./'
elements = ["Si","Cu","Mo","Ni","Ge","Mo","Li"]
mem = []
train_energies = []
train_forces = []
train_stresses = []
train_structures = []
dir_name = "alff2_comb" 
cmd='rm -rf '+dir_name
os.system(cmd)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
for element in elements:
    benchmark_energies = (
        "../jarvis_leaderboard/jarvis_leaderboard/benchmarks/AI/MLFF/mlearn_"
        + element
        + "_energy.json.zip"
    )

    temp_energies = benchmark_energies.split("/")[-1].split(".zip")[0]
    energies = json.loads(
        zipfile.ZipFile(benchmark_energies).read(temp_energies)
    )
    train_ids = list(energies["train"].keys())
    test_ids = list(energies["test"].keys())
    example_config["n_train"] += len(train_ids)
    example_config["n_val"] += len(test_ids)
    example_config["n_test"] += len(test_ids)
    #config_name = dir_name + "/config_" + "comb" + ".json"
    #dumpjson(data=example_config, filename=config_name)

    for i in mlearn:
        if i["jid"] in train_ids:
            # print(i)
            train_energies.append(i["energy"])
            train_forces.append(i["forces"])
            train_stresses.append(i["stresses"])
            atoms = Atoms.from_dict(i["atoms"])
            info = {}
            info["jid"] = i["jid"]
            info["atoms"] = i["atoms"]
            # alignn uses intensive/energy oer atom quanitity
            info["total_energy"] = i["energy"] / atoms.num_atoms
            info["forces"] = i["forces"]
            info["stresses"] = i["stresses"]
            mem.append(info)
    # Val same as test
for element in elements:
    benchmark_energies = (
        "../jarvis_leaderboard/jarvis_leaderboard/benchmarks/AI/MLFF/mlearn_"
        + element
        + "_energy.json.zip"
    )

    temp_energies = benchmark_energies.split("/")[-1].split(".zip")[0]
    energies = json.loads(
        zipfile.ZipFile(benchmark_energies).read(temp_energies)
    )
    train_ids = list(energies["train"].keys())
    test_ids = list(energies["test"].keys())
    for i in mlearn:
        if i["jid"] in test_ids:
            # print(i)
            atoms = Atoms.from_dict(i["atoms"])
            info = {}
            info["jid"] = i["jid"]
            info["atoms"] = i["atoms"]
            # alignn uses intensive/energy oer atom quanitity
            info["total_energy"] = i["energy"] / atoms.num_atoms
            info["forces"] = i["forces"]
            info["stresses"] = i["stresses"]
            mem.append(info)
for element in elements:
    benchmark_energies = (
        "../jarvis_leaderboard/jarvis_leaderboard/benchmarks/AI/MLFF/mlearn_"
        + element
        + "_energy.json.zip"
    )

    temp_energies = benchmark_energies.split("/")[-1].split(".zip")[0]
    energies = json.loads(
        zipfile.ZipFile(benchmark_energies).read(temp_energies)
    )
    train_ids = list(energies["train"].keys())
    test_ids = list(energies["test"].keys())
    for i in mlearn:
        if i["jid"] in test_ids:
            # print(i)
            atoms = Atoms.from_dict(i["atoms"])
            info = {}
            info["jid"] = i["jid"]
            info["atoms"] = i["atoms"]
            # alignn uses intensive/energy oer atom quanitity
            info["total_energy"] = i["energy"] / atoms.num_atoms
            info["forces"] = i["forces"]
            info["stresses"] = i["stresses"]
            mem.append(info)
filename = dir_name + "/id_prop.json"
cfilename = dir_name + "/config_comb.json"
dumpjson(data=mem, filename=filename)
dumpjson(data=example_config, filename=cfilename)
cmd = (
    "train_folder_ff.py --root_dir "
    + dir_name
    + " --config "
    + cfilename
    + " --output_dir "
    + dir_name
)
#cmd="train_folder_ff.py -h"

print(cmd)
#os.system(cmd)
subprocess.call(cmd, stdout=PIPE,shell=True)
    #p1 = Popen(cmd, stdout=PIPE, shell=True)

for element in elements:
################
    model_path = dir_name

    # calc = AlignnAtomwiseCalculator(path=model_path)
    calc = AlignnAtomwiseCalculator(
        path=model_path,
        force_mult_natoms=False,
        force_multiplier=2,
        stress_wt=-4800,
    )



    def get_alignn_forces(atoms):
        energy = 0.0
        forces = np.zeros((atoms.num_atoms, 3))
        stress = np.zeros((3, 3))
        # try:
        ase_atoms = atoms.ase_converter()
        ase_atoms.calc = calc  # M3GNetCalculator(potential=potential)
        forces = np.array(ase_atoms.get_forces())
        energy = ase_atoms.get_potential_energy()
        stress = voigt_6_to_full_3x3_stress(ase_atoms.get_stress())
        # except:
        #  print ('Failed for',atoms)
        #  pass
        return energy, forces, stress

    # df = pd.DataFrame(mdata)
    df = pd.DataFrame(
        json.loads(
            zipfile.ZipFile("../mlearn.json.zip").read(
                "mlearn.json"
            )
        )
    )
    print(df)
    #for i in glob.glob("../../benchmarks/AI/MLFF/*energy*.zip"):
    for i in glob.glob("../jarvis_leaderboard/jarvis_leaderboard/benchmarks/AI/MLFF/*energy*.zip"):
        if "mlearn" in i and element in i:
            fname_e = (
                "AI-MLFF-energy-"
                + i.split("/")[-1].split("_energy.json.zip")[0]
                + "-test-mae.csv"
            )
            fname_f = (
                "AI-MLFF-forces-"
                + i.split("/")[-1].split("_energy.json.zip")[0]
                + "-test-multimae.csv"
            )
            fname_s = (
                "AI-MLFF-stresses-"
                + i.split("/")[-1].split("_energy.json.zip")[0]
                + "-test-multimae.csv"
            )
            f_e = open(fname_e, "w")
            f_f = open(fname_f, "w")
            f_s = open(fname_s, "w")

            f_e.write("id,prediction\n")
            f_f.write("id,prediction\n")
            f_s.write("id,prediction\n")

            print(i)
            dat = json.loads(
                zipfile.ZipFile(i).read(i.split("/")[-1].split(".zip")[0])
            )
            print(dat["test"])
            for key, val in dat["test"].items():
                entry = df[df["jid"] == key]
                atoms = Atoms.from_dict(entry.atoms.values[0])
                # print(key,val,df[df['jid']==key],atoms)
                # energy,forces=get_alignn_forces(atoms)
                energy, forces, stress = get_alignn_forces(atoms)
                print(key, val, energy, atoms.num_atoms)
                line = key + "," + str(energy) + "\n"
                f_e.write(line)
                line = (
                    key
                    + ","
                    + str(";".join(map(str, np.array(forces).flatten())))
                    + "\n"
                )
                f_f.write(line)
                line = (
                    key
                    + ","
                    + str(";".join(map(str, np.array(stress).flatten())))
                    + "\n"
                )
                f_s.write(line)
            f_e.close()
            f_f.close()
            f_s.close()
            zname = fname_e + ".zip"
            with zipfile.ZipFile(zname, "w") as myzip:
                myzip.write(fname_e)

            zname = fname_f + ".zip"
            with zipfile.ZipFile(zname, "w") as myzip:
                myzip.write(fname_f)

            zname = fname_s + ".zip"
            with zipfile.ZipFile(zname, "w") as myzip:
                myzip.write(fname_s)
            # cmd = "zip " + fname_e + ".zip " + fname_e
            # os.system(cmd)
            # cmd = "zip " + fname_f + ".zip " + fname_f
            # os.system(cmd)
            # cmd = "zip " + fname_s + ".zip " + fname_s
            # os.system(cmd)
            # cmd = "rm " + fname_e
            # os.system(cmd)
            # cmd = "rm " + fname_f
            # os.system(cmd)
            # cmd='rm '+fname_s
            # os.system(cmd)
            # break
