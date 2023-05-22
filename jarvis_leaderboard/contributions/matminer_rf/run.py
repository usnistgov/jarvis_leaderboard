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
from jarvis.db.figshare import data, get_request_data
from jarvis.core.atoms import Atoms
import zipfile
import json
import time

tqdm.pandas()

#%%
'''
Define regressor and featurizer
'''

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def get_model(task):
    if task == 'SinglePropertyPrediction':
        model =  RandomForestRegressor
    elif task == 'SinglePropertyClass':
        model =  RandomForestClassifier

    reg = Pipeline([
                ('imputer', SimpleImputer()), 
                ('scaler', StandardScaler()),
                ('model', model(n_estimators=100, max_features=1/3, n_jobs=-1,bootstrap=False,random_state=0))
            ])
    return reg


def to_unitcell(structure):
    '''
    Make sure coordinates are within the unit cell.
    Used before using structural featurizer.

    Parameters
    ----------
    structure :  pymatgen.core.structure.Structure

    Returns
    -------
    structure :  pymatgen.core.structure.Structure
    '''    
    [site.to_unit_cell(in_place=True) for site in structure.sites]
    return structure


def Featurizer(
        df,
        col_id='structure',
        ignore_errors=True,
        chunksize=20
        ):
    """
    Featurize a dataframe using Matminter featurizers

    Parameters
    ----------
    df : Pandas.DataFrame 
        DataFrame with a column named "structure"

    Returns
    -------
    A DataFrame containing labels as the first columns and features as the rest 

    """
    # For featurization
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.composition import (ElementProperty, 
                                                  Stoichiometry, 
                                                  ValenceOrbital, 
                                                  IonProperty)
    from matminer.featurizers.structure import (SiteStatsFingerprint, 
                                                StructuralHeterogeneity,
                                                ChemicalOrdering, 
                                                StructureComposition, 
                                                MaximumPackingEfficiency)   
    # Make sure df is a DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()   
    # Use composition featurizers if inputs are compositions, otherwise use
    # both composition and structure featurizers
    if col_id != 'structure':
        # convert string to composition 
        a = StrToComposition()
        a._overwrite_data = True
        df[col_id] = a.featurize_dataframe(df,col_id,pbar=False)['composition']
        # no structural features
        struc_feat = []
        # 145 compositional features
        compo_feat = [
            Stoichiometry(),
            ElementProperty.from_preset("magpie"),
            ValenceOrbital(props=['frac']),
            IonProperty(fast=True)
            ]
    else:
        # Ensure sites are within unit cells
        df[col_id] = df[col_id].apply(to_unitcell)
        # 128 structural feature
        struc_feat = [
            SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"), 
            SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"),
            StructuralHeterogeneity(),
            MaximumPackingEfficiency(),
            ChemicalOrdering()
            ]       
        # 145 compositional features
        compo_feat = [
            StructureComposition(Stoichiometry()),
            StructureComposition(ElementProperty.from_preset("magpie")),
            StructureComposition(ValenceOrbital(props=['frac'])),
            StructureComposition(IonProperty(fast=True))
            ]
    # Define the featurizer
    featurizer = MultipleFeaturizer(struc_feat+compo_feat)    
    # Set the chunksize used for Pool.map parallelisation
    featurizer.set_chunksize(chunksize=chunksize)
    X = featurizer.featurize_dataframe(df,col_id,ignore_errors=ignore_errors)  
    # check failed entries    
    failed = np.any(pd.isnull(X.iloc[:,df.shape[1]:]), axis=1)
    if np.sum(failed) > 0:
        print(f'Number failed: {np.sum(failed)}/{len(failed)}')
    print('Featurization completed.')
    return X, failed


# get the available properties for the database db
def get_props(db):
    dir = f"../../benchmarks/AI/{task}"
    # get all the files that starts with db and ends with .json.zip in dir
    files = [f for f in os.listdir(dir) if f.startswith(db) and f.endswith(".json.zip")]
    # remove the db name and .json.zip from the file name
    files = [f.replace(db+"_", "").replace(".json.zip", "") for f in files]
    return files 

#%%

task = 'SinglePropertyPrediction' # 'SinglePropertyClass', 'SinglePropertyPrediction',
reg = get_model(task)

special_dbs = ['ssub','supercon_chem','mag2d_chem']

#for db in special_dbs + ['hmof','megnet','qe_tb', 'dft_3d',]: # 'hmof','megnet','qe_tb', 'dft_3d', 
for db in ['qmof',]:

    # Get the whole dataset and featurize for once and for all properties 
        
    if db in special_dbs:
        if db == 'ssub':
            dat = get_request_data(js_tag="ssub.json",url="https://figshare.com/ndownloader/files/40084921")
        elif db == 'supercon_chem':
            dat = get_request_data(js_tag="supercon_chem.json",url="https://figshare.com/ndownloader/files/40719260")
        elif db == 'mag2d_chem':
            dat = get_request_data(js_tag="mag2d_chem.json",url="https://figshare.com/ndownloader/files/40720004")

        n_features = 145
        col_id = 'formula'

    else:
        dat = data(db)
        n_features = 273

    df = pd.DataFrame(dat)

    # some cleaning
    if  db == 'supercon_chem':
        df['formula']= df['formula'].apply(lambda x: x.replace('+',''))
        df['formula']= df['formula'].apply(lambda x: x.replace('-',''))
        df['formula']= df['formula'].apply(lambda x: x.replace('=z',''))
        df['formula']= df['formula'].apply(lambda x: x.replace('!1.5',''))

    # X_file = f"X_{db}.csv"
    X_file = f"../matminer_xgboost/X_{db}.csv"

    if not os.path.exists(X_file):
        # if atomic structure, get the structure
        if db not in special_dbs:
            structure = f'structure_{db}.pkl'
            if os.path.exists(structure):
                df = pd.read_pickle(structure)
            else:           
                df["structure"] = df["atoms"].progress_apply(
                    lambda x: Atoms.from_dict(x).pymatgen_converter()
                )
                df.to_pickle(structure)

        df = df.sample(frac=1, random_state=123)
        X, failed = Featurizer(df,col_id=col_id)
        X.to_csv(X_file)
    
    df = pd.read_csv(X_file)

    for index_name in ['id','jid','SNUMAT_id']:
        if index_name in df.columns:
            df.set_index(index_name,inplace=True)
            df.index = df.index.astype(str)

    for prop in get_props(db):   
        print('')
        print('----------------------------')
        print('')
        print("Running", db, prop)

        fname = f"AI-{task}-{prop}-{db}-test-mae.csv"

        # skip this loop if the file already exists
        if os.path.exists(fname) or os.path.exists(fname + ".zip"):
            print("Benchmark already done, skipping", fname)
            continue

        json_zip = f"../../benchmarks/AI/{task}/{db}_{prop}.json.zip"
        temp2 = f"{db}_{prop}.json"
        zp = zipfile.ZipFile(json_zip)   
        train_val_test = json.loads(zp.read(temp2))

        train = train_val_test["train"]
        if 'val' in train_val_test:
            val = train_val_test["val"]
        else:
            val = {}
        test = train_val_test["test"]

        n_train = len(train)
        n_val = len(val)
        n_test = len(test)

        print("number of training samples", n_train)
        print("number of validation samples", n_val)
        print("number of test samples", n_test)

        ids = list(train.keys()) + list(val.keys()) + list(test.keys())   
        id_test = ids[-n_test:]

        features = df.columns[-n_features:]
        X = df.loc[ids,features]
        y = list(train.values()) + list(val.values()) + list(test.values())
        X = np.array(X)
        y = np.array(y).reshape(-1, 1).astype(np.float64)        

        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[-(n_val + n_test) : -n_test], y[-(n_val + n_test) : -n_test]
        X_test, y_test = X[-n_test:], y[-n_test:]

        t1 = time.time()
        reg.fit(X_train, y_train)        
        pred = reg.predict(X_test)
        t2 = time.time()
        
        # write the predictions to a csv file
        f = open(fname, "w")
        line = "id,prediction\n"
        f.write(line)
        for j, k in zip(id_test, pred):
            line = str(j) + "," + str(k) + "\n"
            f.write(line)
        f.close()
        # zip the csv file
        os.system("zip " + fname + ".zip " + fname)        
        # remove the csv file
        os.remove(fname)

        # print time and metrics
        print("Time", t2 - t1)
        if task == 'SinglePropertyPrediction':
            reg_sc = regr_scores(y_test, pred)
            print(prop, reg_sc["mae"])
        elif task == 'SinglePropertyClass':
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_test, pred)
            print(prop, acc)



