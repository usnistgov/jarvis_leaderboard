"""Modules for making point-defect vacancies."""
from jarvis.analysis.defects.vacancy import Vacancy
import pprint
from jarvis.analysis.thermodynamics.energetics import unary_energy
from collections import OrderedDict
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.core.utils import rand_select
from jarvis.core.atoms import Atoms
import glob
import random
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from jarvis.core.specie import Specie
from jarvis.db.jsonutils import loadjson, dumpjson
from jarvis.db.figshare import data
from jarvis.db.figshare import get_jid_data
from jarvis.analysis.defects.vacancy import Vacancy
from jarvis.analysis.thermodynamics.energetics import unary_energy

# from alignn.pretrained import get_figshare_model
from jarvis.db.figshare import data
from jarvis.analysis.thermodynamics.energetics import get_optb88vdw_energy
import zipfile
import json
import glob
import pandas as pd
import numpy as np
from jarvis.core.atoms import Atoms
import os
import torch
torch.cuda.is_available = lambda : False
# from m3gnet.models import M3GNet, M3GNetCalculator, Potential
# potential = Potential(M3GNet.load())
# calculator = M3GNetCalculator(potential=potential, stress_weight=0.01)
# wget https://figshare.com/ndownloader/files/40357663 -O mlearn.json.zip

from alignn.ff.ff import AlignnAtomwiseCalculator, default_path,wt01_path,wt10_path

# torch.cuda.is_available = lambda : False
model_path = wt10_path()
calculator = AlignnAtomwiseCalculator(path=model_path, stress_wt=0.3)


# from m3gnet.models import M3GNet, M3GNetCalculator, Potential
# potential = Potential(M3GNet.load())
# calculator = M3GNetCalculator(potential=potential, stress_weight=0.01)
# wget https://figshare.com/ndownloader/files/40357663 -O mlearn.json.zip


def atom_to_energy(atoms):
    num_atoms = atoms.num_atoms
    atoms = atoms.ase_converter()
    atoms.calc = calculator
    forces = atoms.get_forces()
    energy = atoms.get_potential_energy()
    stress = atoms.get_stress()
    return energy / num_atoms  # ,forces,stress


unary_data = get_optb88vdw_energy()

#wget https://figshare.com/ndownloader/files/41075822 -O vacancydb.json.zip
#unzip vacancydb.json.zip
dat = loadjson("../alignnff_wt0.1_v1/vacancydb.json")


m = {}
train = {}
test = {}
count = 0
scale=1.0

f = open("AI-SinglePropertyPrediction-ef-vacancydb-test-mae.csv", "w")
f.write("id,target,prediction\n")
for i in dat:
    try:
        count += 1
        # print (i)
        symbol = i["symbol"]
        wycoff = i["wycoff"]
        bulk_atoms = Atoms.from_dict(i["bulk_atoms"])
        defective_atoms = Atoms.from_dict(i["defective_atoms"])
        chem_pot_jid = unary_data[i["symbol"]]["jid"]
        chemo_pot_atoms = Atoms.from_dict(
            get_jid_data(jid=chem_pot_jid, dataset="dft_3d")["atoms"]
        )
        bulk_enp = atom_to_energy(atoms=bulk_atoms)# * bulk_atoms.num_atoms
        def_en = (
            atom_to_energy(atoms=defective_atoms) * defective_atoms.num_atoms
        )
        mu = atom_to_energy(atoms=chemo_pot_atoms)
        Ef = def_en - bulk_enp*(defective_atoms.num_atoms+1) + mu+scale
        name = i["jid"] + "_" + symbol + "_" + wycoff
        line = str(name) + "," + str(i["ef"]) + "," + str(Ef) + "\n"
        # line = str(i["jid"]) + "," + str(i["EF"]) + ","+str(Ef) + "\n"
        f.write(line)
        test[name] = i["ef"]
        print(line)
    except:
        pass
        # break
f.close()

m["train"] = train
m["test"] = test
dumpjson(data=m, filename="vacancydb_ef.json")

cmd = 'zip AI-SinglePropertyPrediction-ef-vacancydb-test-mae.csv.zip AI-SinglePropertyPrediction-ef-vacancydb-test-mae.csv'
os.system(cmd)

df=pd.read_csv('AI-SinglePropertyPrediction-ef-vacancydb-test-mae.csv')
fname='../../benchmarks/AI/SinglePropertyPrediction/vacancydb_oxides_ef.json.zip'


temp='vacancydb_oxides_ef.json'
z = zipfile.ZipFile(fname)
json_data = json.loads(z.read(temp))
list_ids = list(json_data['test'].keys())
new_df = df[df['id'].isin(list_ids)]
csv_name = 'AI-SinglePropertyPrediction-ef-'+temp.split('_ef.json')[0]+'-test-mae.csv'
new_df.to_csv(csv_name,index=False)
cmd='zip '+csv_name+'.zip '+csv_name
os.system(cmd)



fname='../../benchmarks/AI/SinglePropertyPrediction/vacancydb_2D_ef.json.zip'
temp='vacancydb_2D_ef.json'
z = zipfile.ZipFile(fname)
json_data = json.loads(z.read(temp))
list_ids = list(json_data['test'].keys())
new_df = df[df['id'].isin(list_ids)]
csv_name = 'AI-SinglePropertyPrediction-ef-'+temp.split('_ef.json')[0]+'-test-mae.csv'
new_df.to_csv(csv_name,index=False)
cmd='zip '+csv_name+'.zip '+csv_name
os.system(cmd)

fname='../../benchmarks/AI/SinglePropertyPrediction/vacancydb_elements_ef.json.zip'
temp='vacancydb_elements_ef.json'
z = zipfile.ZipFile(fname)
json_data = json.loads(z.read(temp))
list_ids = list(json_data['test'].keys())
new_df = df[df['id'].isin(list_ids)]
csv_name = 'AI-SinglePropertyPrediction-ef-'+temp.split('_ef.json')[0]+'-test-mae.csv'
new_df.to_csv(csv_name,index=False)
cmd='zip '+csv_name+'.zip '+csv_name
os.system(cmd)


