"""Modules for making point-defect vacancies."""
from jarvis.analysis.defects.vacancy import Vacancy
import pprint
from jarvis.analysis.thermodynamics.energetics import unary_energy
from collections import OrderedDict
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.core.utils import rand_select
from jarvis.core.atoms import Atoms
from alignn.pretrained import get_figshare_model
from jarvis.core.graphs import Graph
import glob
import random
from jarvis.db.figshare import data
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.specie import Specie
from jarvis.core.graphs import Graph
from alignn.models.alignn import ALIGNN
from jarvis.db.jsonutils import loadjson, dumpjson
import os
# from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.db.figshare import get_jid_data
from jarvis.analysis.defects.vacancy import Vacancy
from jarvis.analysis.thermodynamics.energetics import unary_energy
from jarvis.db.figshare import data
from jarvis.analysis.thermodynamics.energetics import get_optb88vdw_energy

unary_data = get_optb88vdw_energy()

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
device = "cpu"

import torch
torch.cuda.is_available = lambda : False
print('device',device)
from alignn.pretrained import get_figshare_model
def atom_to_energy(atoms=None, model=None):
    """Get energy for Atoms."""
    g, lg = Graph.atom_dgl_multigraph(atoms)
    out_data = (
        model([g.to(device), lg.to(device)])
        .detach()
        .cpu()
        .numpy()
        .flatten()
        .tolist()[0]
    )
    return out_data


model = get_figshare_model("jv_optb88vdw_total_energy_alignn")
model=model.to('cpu')

# dft_3d = data("dft_3d")
# dft_3d = data("dft_2d")

#dat = loadjson("defect_db_dft.json")
#wget https://figshare.com/ndownloader/files/40750811 -O vacancydb.json.zip
#unzip vacancydb.json.zip
dat = loadjson("../alignnff_wt0.1_v1/vacancydb.json")

scale=1.1
f = open("AI-SinglePropertyPrediction-ef-vacancydb-test-mae.csv", "w")
f.write("id,target,prediction\n")
for i in dat:
 try:
    # print (i)
    bulk_atoms = Atoms.from_dict(i["bulk_atoms"])
    defective_atoms = Atoms.from_dict(i["defective_atoms"])
    chem_pot_jid = unary_data[i["symbol"]]["jid"]
    chemo_pot_atoms = Atoms.from_dict(
        get_jid_data(jid=chem_pot_jid, dataset="dft_3d")["atoms"]
    )
    bulk_enp = (
        atom_to_energy(atoms=bulk_atoms, model=model)# * bulk_atoms.num_atoms
    )
    def_en = (
        atom_to_energy(atoms=defective_atoms, model=model)
        * defective_atoms.num_atoms
    )
    mu = atom_to_energy(atoms=chemo_pot_atoms, model=model)
    Ef = def_en - bulk_enp*(defective_atoms.num_atoms+1) + mu+scale
    symbol = i['symbol']
    wycoff = i['wycoff']
    name = i["jid"] + "_" + symbol + "_" + wycoff
    line = str(name) + "," + str(i["ef"]) + "," + str(Ef) + "\n"
    f.write(line)
    print(line)
 except:
   pass
    # break
f.close()
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

