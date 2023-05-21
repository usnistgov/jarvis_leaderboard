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
#from alignn.pretrained import get_figshare_model
from jarvis.db.figshare import data
from jarvis.analysis.thermodynamics.energetics import get_optb88vdw_energy
import zipfile
import json
import glob
import pandas as pd
import numpy as np
from jarvis.core.atoms import Atoms
import os
#from m3gnet.models import M3GNet, M3GNetCalculator, Potential
#potential = Potential(M3GNet.load())
#calculator = M3GNetCalculator(potential=potential, stress_weight=0.01)
#wget https://figshare.com/ndownloader/files/40357663 -O mlearn.json.zip
from chgnet.model import CHGNet

#wget https://figshare.com/ndownloader/files/40357663 -O mlearn.json.zip


chgnet = CHGNet.load()
def atom_to_energy(atoms):
    num_atoms=atoms.num_atoms
    atoms=atoms.pymatgen_converter()
    prediction = chgnet.predict_structure(atoms)
    energy = prediction['e']
    forces = prediction['f']
    stress = prediction['s']
    return energy/num_atoms #,forces,stress



unary_data = get_optb88vdw_energy()


dat = loadjson("../alignn_pretrained_energy_model/defect_db_dft.json")


scale=5
f = open("prediction_vac.csv", "w")
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
    bulk_en = (
        atom_to_energy(atoms=bulk_atoms) * bulk_atoms.num_atoms
    )
    def_en = (
        atom_to_energy(atoms=defective_atoms)
        * defective_atoms.num_atoms
    )
    mu = atom_to_energy(atoms=chemo_pot_atoms)
    Ef = def_en - bulk_en + mu + 5
    line = str(i["jid"]) + "," + str(i["EF"]) + ","+str(Ef) + "\n"
    f.write(line)
    print(line)
 except:
   pass
    # break
f.close()
# x=[]
# for i in glob.glob("*.json"):
#  x.append(i.split('.json')[0])
#
# for i in dft_3d:
# if i["jid"] not in x:
#    try:
#        mem = jid_ef(
#            model=model, jid=i["jid"]
#        )  # atoms=Atoms.from_dict(dft_3d[0]['atoms']))
#    except:
#        pass
