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

torch.cuda.is_available = lambda: False
from alignn.ff.ff import AlignnAtomwiseCalculator, default_path, wt01_path

# torch.cuda.is_available = lambda : False
model_path = wt01_path()
calculator = AlignnAtomwiseCalculator(path=model_path, stress_wt=0.3)

def atom_to_energy(atoms):
    num_atoms = atoms.num_atoms
    atoms = atoms.ase_converter()
    atoms.calc = calculator
    forces = atoms.get_forces()
    energy = atoms.get_potential_energy()
    stress = atoms.get_stress()
    return energy / num_atoms  # ,forces,stress


unary_data = get_optb88vdw_energy()

# wget https://figshare.com/ndownloader/files/41075822 -O vacancydb.json.zip
# unzip vacancydb.json.zip
dat = loadjson("vacancydb3.json")


m = {}
train = {}
test = {}
count = 0
scale = 0.8
from jarvis.analysis.thermodynamics.energetics import unary_energy

f = open("AI-SinglePropertyPrediction-ef-vacancydb-test-mae.csv", "w")
f.write("id,target,prediction,prediction_ini,id_ini\n")
for i in dat:
    try:
        # print (i)
        name = i["id"]
        # name = i['id'].split('_')[0]+'_'+i['id'].split('_')[1]+'_'+i['id'].split('_')[2]
        bulk_atoms = Atoms.from_dict(i["vacancy_class"]["atoms"])
        defective_atoms_ini = Atoms.from_dict(
            i["vacancy_class"]["defect_structure"]
        )
        defective_atoms = Atoms.from_dict(i["defective_atoms"])
        mu = unary_energy(
            el=i["vacancy_class"]["symbol"]
        )  # unary_data[i["symbol"]]["jid"]
        bulk_enp = atom_to_energy(atoms=bulk_atoms)  # * bulk_atoms.num_atoms
        def_en_relax = (
            atom_to_energy(atoms=defective_atoms) * defective_atoms.num_atoms
        )
        def_en_ini = (
            atom_to_energy(atoms=defective_atoms_ini)
            * defective_atoms.num_atoms
        )
        Ef_ini = (
            def_en_ini
            - bulk_enp * (defective_atoms.num_atoms + 1)
            + mu
            + scale
        )
        Ef = (
            def_en_relax
            - bulk_enp * (defective_atoms.num_atoms + 1)
            + mu
            + scale
        )
        line = (
            str(name)
            + ","
            + str(i["ef"])
            + ","
            + str(Ef)
            + ","
            + str(Ef_ini)
            + ","
            + str(i["id"])
            + "\n"
        )
        # line = str(i["jid"]) + "," + str(i["EF"]) + ","+str(Ef) + "\n"
        print(line)
        f.write(line)
        test[name] = i["ef"]
    except:
        print("Failed for", name)
        pass
        # break
f.close()


"""
df=pd.read_csv('AI-SinglePropertyPrediction-ef-vacancydb-test-mae.csv')
df['ele']=df['id'].apply(lambda x:x.split('_')[1])
dfO=df[df['ele']=='O']
m={}
m['train']={}
info={}
for i,ii in df.iterrows():
   info[ii['id']]=ii['target']
m['test']=info
dumpjson(data=m,filename='vacancydb_ef.json')

m={}
m['train']={}
info={}
for i,ii in dfO.iterrows():
   info[ii['id']]=ii['target']
m['test']=info
dumpjson(data=m,filename='vacancydb_oxides_ef.json')

jids=[]
pp=loadjson('vacancydb2.json')
for i in pp:
  if i['material_class']=='2D':
      jids.append(i['id'])
dfO.to_csv('AI-SinglePropertyPrediction-ef-vacancydb_oxides-test-mae.csv',index=False)
df2d=df[df['id'].isin(jids)]

m={}
m['train']={}
info={}
for i,ii in df2d.iterrows():
   info[ii['id']]=ii['target']
m['test']=info
dumpjson(data=m,filename='vacancydb_2D_ef.json')

df2d.to_csv('AI-SinglePropertyPrediction-ef-vacancydb_2D-test-mae.csv',index=False)

x=get_optb88vdw_energy()
df['jid']=df['id'].apply(lambda x: x.split('_')[0])
els=[i['jid'] for i in list(x.values())]
df_els = df[df['jid'].isin(els)]
df_els.to_csv('AI-SinglePropertyPrediction-ef-vacancydb_elements-test-mae.csv',index=False)

m={}
m['train']={}
info={}
for i,ii in df_els.iterrows():
   info[ii['id']]=ii['target']
m['test']=info
dumpjson(data=m,filename='vacancydb_elements_ef.json')


"""

"""
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


"""
