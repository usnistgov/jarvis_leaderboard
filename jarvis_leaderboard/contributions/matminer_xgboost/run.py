"""Module to run matminer results."""
#%%
import random
import os
import shutil
import pandas as pd
from tqdm import tqdm
import csv
import numpy as np
import math
from jarvis.ai.pkgs.utils import regr_scores
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
import zipfile
import json
import time

tqdm.pandas()

n_features = 273

task = 'SinglePropertyPrediction'


#%%
'''
Define regressor and featurizer
'''

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb



#%%
'''
Model
'''

n_estimators = 10000
num_parallel_tree = 8
learning_rate = 0.1  
tree_method = 'gpu_hist'  
reg = pipe = Pipeline([
            ('imputer', SimpleImputer()), 
            ('scaler', StandardScaler()),
            ('model', xgb.XGBRegressor(
                            n_estimators=n_estimators, learning_rate=learning_rate,
                            reg_lambda=0.01,reg_alpha=0.01,
                            subsample=0.85,colsample_bytree=0.3,colsample_bylevel=0.5,
                            num_parallel_tree=num_parallel_tree,
                            tree_method=tree_method
                            ))
        ])

#%%
# https://github.com/mathsphy/paper-ml-robustness-material-property/blob/main/myfunc.py
def StructureFeaturizer(
    df_in, col_id="structure", ignore_errors=True, chunksize=30
):
    """
    Featurize a dataframe using Matminter Structure featurizer

    Parameters
    ----------
    df : Pandas.DataFrame
        DataFrame with a column named "structure"

    Returns
    -------
    A DataFrame containing 273 features (columns)

    """
    # For featurization
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (
        ElementProperty,
        Stoichiometry,
        ValenceOrbital,
        IonProperty,
    )
    from matminer.featurizers.structure import (
        SiteStatsFingerprint,
        StructuralHeterogeneity,
        ChemicalOrdering,
        StructureComposition,
        MaximumPackingEfficiency,
    )

    if isinstance(df_in, pd.Series):
        df = df_in.to_frame()
    else:
        df = df_in
    # df[col_id] = df[col_id].apply(to_unitcell)

    # 128 structural feature
    struc_feat = [
        SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"),
        SiteStatsFingerprint.from_preset(
            "LocalPropertyDifference_ward-prb-2017"
        ),
        StructuralHeterogeneity(),
        MaximumPackingEfficiency(),
        ChemicalOrdering(),
    ]
    # 145 compositional features
    compo_feat = [
        StructureComposition(Stoichiometry()),
        StructureComposition(ElementProperty.from_preset("magpie")),
        StructureComposition(ValenceOrbital(props=["frac"])),
        StructureComposition(IonProperty(fast=True)),
    ]
    featurizer = MultipleFeaturizer(struc_feat + compo_feat)

    # Set the chunksize used for Pool.map parallelisation
    featurizer.set_chunksize(chunksize=chunksize)
    featurizer.fit(df[col_id])
    X = featurizer.featurize_dataframe(
        df=df, col_id=col_id, ignore_errors=ignore_errors
    )
    # check failed entries
    print("Featurization completed.")
    failed = np.any(pd.isnull(X.iloc[:, df.shape[1] :]), axis=1)
    if np.sum(failed) > 0:
        print(f"Number failed: {np.sum(failed)}/{len(failed)}")
    return X, failed

def load_dataset(
    name,
    df = None,
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
    if df is None:
        df = pd.read_csv(f"X_{name}.csv")
    important_features = df.columns[-n_features:]

    id_prop_dat = os.path.join(root_dir, "id_prop.csv")
    with open(id_prop_dat, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    dataset = []
    n_outputs = []
    multioutput = False
    lists_length_equal = True

    def get_desc(id=""):
        return (df[df["jid"] == id][important_features]).values[0]

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




#%%    
# get the available properties for the database db
def get_props(db):
    dir = f"../../benchmarks/AI/{task}"
    # get all the files that starts with db and ends with .json.zip in dir
    files = [f for f in os.listdir(dir) if f.startswith(db) and f.endswith(".json.zip")]
    # remove the db name and .json.zip from the file name
    files = [f.replace(db+"_", "").replace(".json.zip", "") for f in files]
    return files 

#%%
for db in ['dft_3d',]: #'hmof','qm9','megnet','qe_tb',

    # Get the whole dataset and featurize for once and for all properties 
    dat = data(db)
    X_file = f"X_{db}.csv"
    if not os.path.exists(X_file):
        df = pd.DataFrame(dat)
        df["structure"] = df["atoms"].progress_apply(
            lambda x: (
                (Atoms.from_dict(x)).get_primitive_atoms
            ).pymatgen_converter()
        )

        df = df.sample(frac=1, random_state=123)
        X, failed = StructureFeaturizer(df)
        X.to_csv(X_file)

    df = pd.read_csv(X_file)


    for prop in get_props(db):   

        fname = f"AI-{task}-{prop}-{db}-test-mae.csv"

        json_zip = f"../../benchmarks/AI/{task}/{db}_{prop}.json.zip"

        # skip this loop if the file already exists
        if os.path.exists(fname) or os.path.exists(fname + ".zip"):
            print("Benchmark already done, skipping", fname)
            continue
        elif not os.path.exists(json_zip):
            print("Benchmark not exists, skipping", fname)
            continue

        temp2 = f"{db}_{prop}.json"
        zp = zipfile.ZipFile(json_zip)   
        train_val_test = json.loads(zp.read(temp2))

        output_path = "DataDir-" + prop
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
        
        id_tag = "jid"
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

        t1 = time.time()
        X_train, y_train, X_val, y_val, X_test, y_test, id_test = load_dataset(
            name=db,
            df=df,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            root_dir=output_path,
            target=prop,
        )

        reg.fit(X_train, y_train)
        
        pred = reg.predict(X_test)
        reg_sc = regr_scores(y_test, pred)
        print(prop, reg_sc["mae"])

        f = open(fname, "w")
        line = "id,prediction\n"
        f.write(line)
        for j, k in zip(id_test, pred):
            line = str(j) + "," + str(k) + "\n"
            f.write(line)
        f.close()
        t2 = time.time()
        print("Time", t2 - t1)
        cmd = "zip " + fname + ".zip " + fname
        os.system(cmd)
        # remove output_path
        shutil.rmtree(output_path)
