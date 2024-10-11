from alignn.graphs import Graph
from alignn.pretrained import get_figshare_model
from tqdm import tqdm
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
import pandas as pd
import os, torch
from sklearn.metrics import mean_absolute_error, r2_score
from jarvis_leaderboard.rebuild import get_metric_value, get_results
from jarvis.core.composition import Composition
from jarvis.core.atoms import Atoms
from scipy.stats import pearsonr
from jarvis.analysis.thermodynamics.energetics import get_optb88vdw_energy
from alignn.ff.ff import (
    AlignnAtomwiseCalculator,
    default_path,
    wt1_path,
    wt01_path,
    get_figshare_model_ff,
)
from jarvis.db.jsonutils import loadjson, dumpjson
import zipfile
import os

def zip_file(file_path, zip_name):
    # Create a zip file with the provided name
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        # Add the file to the zip archive
        zipf.write(file_path, os.path.basename(file_path))
    print(f'{file_path} has been zipped as {zip_name}')


dft_3d=data('dft_3d')

def get_entry(jid):
    for i in dft_3d:
        if i["jid"] == jid:
            return i
mu = get_optb88vdw_energy()
# torch.cuda.is_available = lambda : False
# dpath = get_figshare_model_ff(model_name="scf_fd_top_10_en_42_fmax_600_wt01")
model_path = wt01_path()  # wt1_path()
model_path='/work/03943/kamalch/ls6/AFF_Bench/aff307k_lmdb_param_low_rad_use_force_mult_mp_tak4_cut4/out111/'
model_path='/work/03943/kamalch/ls6/AFF_Bench/aff307k_lmdb_param_low_rad_use_cutoff_take4_noforce_mult_cut4/out111a'
calculator = AlignnAtomwiseCalculator(path=model_path, stress_wt=0.3,filename='current_model.pt')

import matgl
from matgl.ext.ase import M3GNetCalculator


pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
calculator = M3GNetCalculator(pot, stress_weight=0.01)

"""

from chgnet.model.dynamics import CHGNetCalculator
calculator=CHGNetCalculator()


#from mace.calculators import mace_mp
#calculator=mace_mp()
"""
def atom_to_energy(atoms):
    num_atoms = atoms.num_atoms
    atoms = atoms.ase_converter()
    atoms.calc = calculator
    forces = atoms.get_forces()
    energy = atoms.get_potential_energy()
    stress = atoms.get_stress()
    return energy  # ,forces,stress


def remove_atom(atoms=[], exclude_elements=["O"]):
    coords = []
    elements = []
    for i, j in zip(atoms.elements, atoms.cart_coords):
        if i not in exclude_elements:
            elements.append(i)
            coords.append(j)

    atoms = Atoms(
        lattice_mat=atoms.lattice_mat,
        elements=elements,
        coords=coords,
        cartesian=True,
    )
    return atoms


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
# res = get_metric_value(
#     csv_path="jarvis_leaderboard/contributions/alignn_model/AI-SinglePropertyPrediction-ead-AGRA_CO-test-mae.csv.zip"
# )
models = [
    "uMLff",
    #"ocp2020_all",
]
benchmarks = [
    "AI-SinglePropertyPrediction-ead-tinnet_O-test-mae.csv.zip",
    "AI-SinglePropertyPrediction-ead-tinnet_N-test-mae.csv.zip",
    "AI-SinglePropertyPrediction-ead-tinnet_OH-test-mae.csv.zip",
    "AI-SinglePropertyPrediction-ead-AGRA_O-test-mae.csv.zip",
    "AI-SinglePropertyPrediction-ead-AGRA_CHO-test-mae.csv.zip",
    "AI-SinglePropertyPrediction-ead-AGRA_CO-test-mae.csv.zip",
    "AI-SinglePropertyPrediction-ead-AGRA_COOH-test-mae.csv.zip",
    "AI-SinglePropertyPrediction-ead-AGRA_OH-test-mae.csv.zip",
]
#benchmarks = [
#    "AI-SinglePropertyPrediction-ead-tinnet_O-test-mae.csv.zip",
#]
ocp_energy = {"H": -3.477, "O": -7.204, "C": -7.282, "N": -8.083}
mem = []
id_tag = "id"
for ib in benchmarks:
    # if 'tinnet' in ib:
    bn = os.path.abspath(
        os.path.join("/work/03943/kamalch/ls6/Software/alignn_elast/jarvis_leaderboard/jarvis_leaderboard", "contributions", "alignn_model", ib)
    )
    res = get_metric_value(csv_path=bn)
    out_file=ib.split('.zip')[0]
    ff=open(out_file,'w') 
    ff.write('id,target,prediction\n')
    for jm in models:
        try:
            if jm != "uMLff":
              model = get_figshare_model(jm)

              model = model.to(device)
              model.eval()
            # if jm!='aff':
            #      model=get_figshare_model(jm)
            # else:

            dataset = ib.split("-")[3]
            prop = ib.split("-")[2]
            # print('data',dataset)
            dat = pd.DataFrame(data(dataset))
            pd_merged = pd.merge(res["df"], dat, on=id_tag)

            x = []
            y = []
            z = []
            ids = []
            for i, ii in pd_merged.iterrows():
                id = ii["id"]
                atoms = Atoms.from_dict(ii["atoms"])
                g, lg = Graph.atom_dgl_multigraph(atoms)
                g = g.to(device)
                lg = lg.to(device)
                mol = dataset.split("_")[-1]
                comp = Composition.from_string(mol).to_dict()
                rem_atoms = remove_atom(atoms=atoms, exclude_elements=mol)
                if jm != "uMLff": #ALIGNN OCP pretrained model
                    #model = get_figshare_model(jm)

                    #model = model.to(device)
                    #model.eval()
                    pred = model([g, lg]).detach().cpu().numpy().tolist()
                    if "ocp" in jm:
                        chempot = 0.0
                        mol = []
                        for kk, vv in comp.items():
                            # print(kk,vv)
                            chempot += ocp_energy[kk] * vv
                            mol.append(kk)

                        g, lg = Graph.atom_dgl_multigraph(rem_atoms)
                        g = g.to(device)
                        lg = lg.to(device)
                        pred_rem = (
                            model([g, lg]).detach().cpu().numpy().tolist()
                        )
                        # print ('xxx',mol,chempot,pred*atoms.num_atoms,pred_rem*rem_atoms.num_atoms)
                        # pred=pred*atoms.num_atoms-pred_rem*rem_atoms.num_atoms+chempot
                        pred = pred - pred_rem + chempot
                else:
                    #print('jm',jm)
                    chempot = 0.0
                    mol = []
                    for kk, vv in comp.items():
                        # print(kk,vv)
                        jid_elemental = mu[kk]["jid"]
                        atoms_elemental = Atoms.from_dict(get_entry(jid_elemental)["atoms"])
                        en_elemental = atom_to_energy(atoms_elemental)/atoms_elemental.num_atoms
                        chempot += en_elemental* vv
                        #chempot += mu[kk]["energy"] * vv
                        mol.append(kk)
                    e_all_atoms = atom_to_energy(atoms)
                    e_surface = atom_to_energy(rem_atoms)
                    pred = e_all_atoms - e_surface + chempot

                # print(id,pred,ii['prediction'],ii['target'])
                x.append(ii["target"])
                y.append(ii["prediction"])
                z.append(pred)
                ids.append(id)
                print(id,ii["target"],ii["prediction"])
            #out_file=ib.split('.zip')[0]
            #ff=open(out_file,'w') 
            #ff.write('id,target,prediction\n')
            for xx,yy,zz in zip(ids,x,z):
               line=str(xx)+','+str(yy)+','+str(zz)+'\n'
               ff.write(line)
            #ff.close()  
            mae_old = mean_absolute_error(x, y)
            mae_new = mean_absolute_error(x, z)
               #for m,n,p in zip(
               
            pr = pearsonr(x, z)[0]
            r2 = r2_score(x, z)
            print("R2=",ib,  r2)
            print("pR=", ib, pr)
            info = {}
            info["model"] = jm
            info["bench"] = ib
            info["len"] = len(x)
            info["mae_old"] = mae_old
            info["mae_new"] = mae_new
            info["pr"] = pr
            info["R2"] = r2
            info["x"] = x
            info["y"] = y
            info["z"] = z
            mem.append(info)
            #print(mem[-1])
        except Exception as exp:
            print("Error", ib, jm, exp)
            pass

    ff.close()
    zip_file(out_file,ib)
dumpjson(data=mem, filename="mem.json")
# break

