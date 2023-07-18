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

# torch.cuda.is_available = lambda : False
from matgl.ext.ase import M3GNetCalculator
import matgl
#torch.cuda.is_available = lambda : False

pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
calc = M3GNetCalculator(pot)

# export CUDA_VISIBLE_DEVICES=""
# wget https://figshare.com/ndownloader/files/40357663 -O mlearn.json.zip


def get_alignn_forces(atoms, rescale_factor=2.5):
    energy = 0.0
    forces = np.zeros((atoms.num_atoms, 3))
    stress = np.zeros((3, 3))
    # try:
    ase_atoms = atoms.ase_converter()
    ase_atoms.calc = calc  # M3GNetCalculator(potential=potential)
    forces = np.array(ase_atoms.get_forces())
    energy = ase_atoms.get_potential_energy()[0]
    stress = voigt_6_to_full_3x3_stress(ase_atoms.get_stress())
    # except:
    #  print ('Failed for',atoms)
    #  pass
    return energy, forces, stress


df = pd.DataFrame(
    json.loads(
        zipfile.ZipFile("../alignnff_wt1_mlearn_only1/mlearn.json.zip").read(
            "mlearn.json"
        )
    )
)
print(df)
for i in glob.glob("../../benchmarks/AI/MLFF/*energy*.zip"):
    if "mlearn" in i:  # and "Si" in i:
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
