import os
#cmd='conda activate /work/03943/kamalch/ls6/Software/alignn'
#os.system(cmd)
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
def x(element='Si',dir_name='alff_Si'):
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
            zipfile.ZipFile("mlearn.json.zip").read(
                "mlearn.json"
            )
        )
    )
    print(df)
    #for i in glob.glob("../../benchmarks/AI/MLFF/*energy*.zip"):
    for i in glob.glob("jarvis_leaderboard/jarvis_leaderboard/benchmarks/AI/MLFF/*energy*.zip"):
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

x(element='Mo',dir_name='alff2_wt_0.2_Mo')
x(element='Cu',dir_name='alff2_wt_0.2_Cu')
x(element='Ge',dir_name='alff2_wt_0.2_Ge')
x(element='Si',dir_name='alff2_wt_0.2_Si')
x(element='Ni',dir_name='alff2_wt_0.2_Ni')
x(element='Li',dir_name='alff2_wt_0.2_Li')
cmd='cp *.csv*.zip jarvis_leaderboard/jarvis_leaderboard/contributions/alignnff_mlearn_ls6_wt0.2'
os.system(cmd)
#x()
