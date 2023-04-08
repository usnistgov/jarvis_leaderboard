"""Module to compare JARVIS-CFID results."""
# from jarvis.ai.pkgs.utils import get_ml_data
# from jarvis.ai.pkgs.utils import regr_scores
from jarvis.db.figshare import data as jdata
import random
import os
import pandas as pd
from tqdm import tqdm
import csv
import numpy as np
import math
import lightgbm as lgb
from jarvis.ai.pkgs.utils import regr_scores
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
import zipfile
import json
import time
props = [
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "mbj_bandgap",
    "slme",
    "magmom_oszicar",
    "spillage",
    "kpoint_length_unit",
    "encut",
    "optb88vdw_total_energy",
    "epsx",
    "epsy",
    "epsz",
    "mepsx",
    "mepsy",
    "mepsz",
    "max_ir_mode",
    "avg_elec_mass",
    "avg_hole_mass",
    "max_efg",
    "min_ir_mode",
    "n-Seebeck",
    "p-Seebeck",
    "n-powerfact",
    "p-powerfact",
    "ncond",
    "pcond",
    "nkappa",
    "pkappa",
    "ehull",
    "exfoliation_energy",
    "dfpt_piezo_max_dielectric",
    "dfpt_piezo_max_eij",
    "dfpt_piezo_max_dij",
]
#props = ["exfoliation_energy"]


def load_dataset(
    name: str = "cfid_3d",
    root_dir="",
    file_format="poscar",
    target=None,
    n_train=None,
    n_val=None,
    n_test=None,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    split_seed=123,
):
    id_prop_dat = os.path.join(root_dir, "id_prop.csv")
    from jarvis.db.figshare import data

    df = pd.DataFrame(data(name))
    with open(id_prop_dat, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    dataset = []
    n_outputs = []
    multioutput = False
    lists_length_equal = True

    def get_desc(id=""):
        return (df[df["jid"] == id]["desc"]).values[0]

    X = []
    y = []
    ids = []

    for i in tqdm(data):
        info = {}
        file_name = i[0]
        file_path = os.path.join(root_dir, file_name)
        if file_format == "poscar":
            atoms = Atoms.from_poscar(file_path)
        elif file_format == "cif":
            atoms = Atoms.from_cif(file_path)
        elif file_format == "xyz":
            # Note using 500 angstrom as box size
            atoms = Atoms.from_xyz(file_path, box_size=500)
        elif file_format == "pdb":
            # Note using 500 angstrom as box size
            # Recommended install pytraj
            # conda install -c ambermd pytraj
            atoms = Atoms.from_pdb(file_path, max_lat=500)
        else:
            raise NotImplementedError(
                "File format not implemented", file_format
            )

        info["atoms"] = atoms.to_dict()
        info["jid"] = file_name

        tmp = [float(j) for j in i[1:]]  # float(i[1])
        if len(tmp) == 1:
            tmp = tmp[0]
        else:
            multioutput = True
        info["target"] = tmp  # float(i[1])
        n_outputs.append(info["target"])
        dataset.append(info)
        X.append(get_desc(file_name))
        y.append(tmp)
        ids.append(file_name)
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    id_test = ids[-n_test:]

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[-(n_val + n_test) : -n_test]
    y_val = y[-(n_val + n_test) : -n_test]

    X_test = X[-n_test:]
    y_test = y[-n_test:]

    return X_train, y_train, X_val, y_val, X_test, y_test, id_test


lgbm = lgb.LGBMRegressor(
    # device="gpu",
    n_estimators=1740,
    learning_rate=0.040552334327414057,
    num_leaves=291,
    max_depth=16,
    min_data_in_leaf=14,
)

lgbm = lgb.LGBMRegressor(
    # device="gpu",
    n_estimators=1170,
    learning_rate=0.15375236057119931,
    num_leaves=273,
)

prop = ["exfoliation_energy"]
#lgbm = lgb.LGBMRegressor()

for prop in props:
    json_zip = '../../dataset/AI/SinglePropertyPrediction/dft_3d_'+prop+'.json.zip'
    temp2 = 'dft_3d' + "_" + prop + ".json"
    zp = zipfile.ZipFile(json_zip)
    train_val_test = json.loads(zp.read(temp2))
    output_path = 'DataDir-'+prop
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
    info = {}
    dat = data('dft_3d')
    id_tag='jid'
    for i in dat:
       info[i[id_tag]] = Atoms.from_dict(i["atoms"])
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
    n_train = len(train)
    n_val = len(val)
    n_test = len(test)

    print("number of training samples", len(train))
    print("number of validation samples", len(val))
    print("number of test samples", len(test))

    fname = "SinglePropertyPrediction-test-" + prop + "-dft_3d-AI-mae.csv"
    t1 = time.time()
    X_train, y_train, X_val, y_val, X_test, y_test, id_test = load_dataset(
        name="cfid_3d",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        root_dir=output_path,
        target=prop,
    )

    lgbm.fit(X_train, y_train)
    pred = lgbm.predict(X_test)
    reg_sc = regr_scores(y_test, pred)
    print(prop, reg_sc["mae"])

    f = open(fname, "w")
    line = "id,prediction\n"
    f.write(line)
    for j, k in zip(id_test, pred):
        line =  str(j) + "," + str(k) + "\n"
        f.write(line)
    f.close()
    t2 = time.time()
    print("Time", t2 - t1)
    cmd = 'zip '+fname+'.zip '+fname
    os.system(cmd)
