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
tree_method = 'hist'   # gpu_hist or hist
reg = pipe = Pipeline([
            ('imputer', SimpleImputer()), 
            ('scaler', StandardScaler()),
            ('model', xgb.XGBRegressor(
                            # n_jobs=-1, random_state=0,
                            n_estimators=n_estimators, learning_rate=learning_rate,
                            reg_lambda=0.01,reg_alpha=0.01,
                            subsample=0.85,colsample_bytree=0.3,colsample_bylevel=0.5,
                            num_parallel_tree=num_parallel_tree,
                            tree_method=tree_method,
                            ))
        ])

#%%

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

# https://github.com/mathsphy/paper-ml-robustness-material-property/blob/main/myfunc.py
def StructureFeaturizer(
    df_in, col_id="structure", ignore_errors=True, chunksize=35, index_ids=None
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
    
    # if not provided, use all the ids in the dataframe
    if index_ids is None:
        index_ids = df_in.index


    if isinstance(df_in, pd.Series):
        df = df_in.loc[index_ids].copy().to_frame()
    else:
        df = df_in.loc[index_ids].copy()
    

    # can we apply this to QM9?
    df[col_id] = df[col_id].apply(to_unitcell)

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
for db in ['hmof', ]: #'hmof','qm9','megnet','qe_tb', 'dft_3d',

    # Get the whole dataset and featurize for once and for all properties 
    dat = data(db)
    X_file = f"X_{db}.csv"
    if not os.path.exists(X_file):
        structure = f'structure_{db}.pkl'
        if os.path.exists(structure):
            df = pd.read_pickle(structure)
        else:
            
            df = pd.DataFrame(dat)
            df["structure"] = df["atoms"].progress_apply(
                lambda x: (
                    (Atoms.from_dict(x)).get_primitive_atoms
                ).pymatgen_converter()
            )
            df.to_pickle(structure)

        df = df.sample(frac=1, random_state=123)

        for i,df_ in enumerate(np.array_split(df, 10)):
            X_, failed = StructureFeaturizer(df_.copy(),chunksize=5)
            X_.to_csv(f"X_{db}_{i}.pkl")
            X.append(X_)

        X_all = pd.concat(X)
        X_all.to_csv(f"X_{db}.pkl")

