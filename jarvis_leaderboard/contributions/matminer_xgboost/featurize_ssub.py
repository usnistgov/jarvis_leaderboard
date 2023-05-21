#%%
import pandas as pd
import numpy as np
# read dataframe from json file
# json file from https://figshare.com/articles/dataset/ssub_formula_energy_dataset/22583677
df = pd.read_json('ssub.json').set_index('id')
df.columns = ['formula_energy',	'composition']

#%%


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
    if col_id == 'composition':
        # convert string to composition 
        a = StrToComposition()
        a._overwrite_data = True
        df[col_id] = a.featurize_dataframe(df,col_id,pbar=False)[col_id]
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



# %%
X, failed = Featurizer(df,col_id='composition')
# %%
X.to_csv('X_ssub.csv')

# %%
