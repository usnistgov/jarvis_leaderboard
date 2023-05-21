#!/usr/bin/env python
# coding: utf-8

# # 1.SSUB

# In[1]:


#https://github.com/wolverton-research-group/qmpy/blob/master/qmpy/data/thermodata/ssub.dat
from jarvis.db.figshare import  get_request_data
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from pymatgen.core.composition import Composition
from sklearn.model_selection import train_test_split
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (ElementProperty, 
                                              Stoichiometry, 
                                              ValenceOrbital, 
                                              IonProperty)

dat = get_request_data(js_tag="ssub.json",url="https://figshare.com/ndownloader/files/40084921")
df = pd.DataFrame(dat)



# In[2]:


import json,zipfile
import numpy as np
path = "../../benchmarks/AI/SinglePropertyPrediction/ssub_formula_energy.json.zip"
js_tag = "ssub_formula_energy.json"
id_data = json.loads(zipfile.ZipFile(path).read(js_tag))
train_ids = np.array(list(id_data['train'].keys()),dtype='int')
test_ids = np.array(list(id_data['test'].keys()),dtype='int')
train_df = df[df['id'].isin(train_ids)]
test_df = df[df['id'].isin(test_ids)]


# In[3]:


df


# In[4]:


compo_feat = [
    (Stoichiometry()),
    (ElementProperty.from_preset("magpie")),
    (ValenceOrbital(props=['frac'])),
    (IonProperty(fast=True))
    ]
featurizer = MultipleFeaturizer(compo_feat)   
prop = 'formula_energy'
train_df['pmg_comp']=train_df['formula'].apply(lambda x:Composition(x))
ignore_errors=False
X = featurizer.featurize_dataframe(df=train_df,col_id='pmg_comp',ignore_errors=ignore_errors)
train_X_df = X.drop(columns=['id','formula',prop,'pmg_comp'])
train_y_df = X[prop]

test_df['pmg_comp']=test_df['formula'].apply(lambda x:Composition(x))
ignore_errors=False
X = featurizer.featurize_dataframe(df=test_df,col_id='pmg_comp',ignore_errors=ignore_errors)
test_X_df = X.drop(columns=['id','formula',prop,'pmg_comp'])
test_y_df = X[prop]


# In[5]:


get_ipython().run_cell_magic('time', '', 'lgbm = lgb.LGBMRegressor(\n    # device="gpu",\n    n_estimators=1170,\n    learning_rate=0.15,\n    num_leaves=273,\n)\nlgbm.fit(train_X_df,train_y_df)\npred=lgbm.predict(test_X_df)\nprint (mean_absolute_error(test_y_df,pred))\n')


# In[6]:


import os
f=open('AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv','w')
f.write('id,target,prediction\n')
for i,j,k in zip(test_ids,test_y_df.values,pred):
    line=str(i)+','+str(j)+','+str(k)+'\n'
    f.write(line)
f.close()
cmd = 'zip AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv.zip AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv'
os.system(cmd)
cmd='rm AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv'
os.system(cmd)


# In[ ]:





# In[ ]:





# # 2. Mag2D

# In[1]:


get_ipython().run_cell_magic('time', '', '\nfrom jarvis.db.figshare import  get_request_data\nimport pandas as pd\nimport os\nimport lightgbm as lgb\nimport json,zipfile\nimport numpy as np\nfrom sklearn.metrics import mean_absolute_error\nfrom pymatgen.core.composition import Composition\nfrom sklearn.model_selection import train_test_split\nfrom matminer.featurizers.base import MultipleFeaturizer\nfrom matminer.featurizers.composition import (ElementProperty, \n                                              Stoichiometry, \n                                              ValenceOrbital, \n                                              IonProperty)\n\ndat = get_request_data(js_tag="mag2d_chem.json",url="https://figshare.com/ndownloader/files/40720004")\ndf = pd.DataFrame(dat)\n\npath = "../../benchmarks/AI/SinglePropertyPrediction/mag2d_chem_magnetic_moment.json.zip"\njs_tag = "mag2d_chem_magnetic_moment.json"\nid_data = json.loads(zipfile.ZipFile(path).read(js_tag))\ntrain_ids = np.array(list(id_data[\'train\'].keys()),dtype=\'int\')\ntest_ids = np.array(list(id_data[\'test\'].keys()),dtype=\'int\')\ntrain_df = df[df[\'id\'].isin(train_ids)]\ntest_df = df[df[\'id\'].isin(test_ids)]\ncompo_feat = [\n    (Stoichiometry()),\n    (ElementProperty.from_preset("magpie")),\n    (ValenceOrbital(props=[\'frac\'])),\n    (IonProperty(fast=True))\n    ]\nfeaturizer = MultipleFeaturizer(compo_feat)   \nprop = \'magnetic_moment\'\ntrain_df[\'pmg_comp\']=train_df[\'formula\'].apply(lambda x:Composition(x))\nignore_errors=False\nX = featurizer.featurize_dataframe(df=train_df,col_id=\'pmg_comp\',ignore_errors=ignore_errors)\ntrain_X_df = X.drop(columns=[\'id\',\'formula\',prop,\'pmg_comp\'])\ntrain_y_df = X[prop]\n\ntest_df[\'pmg_comp\']=test_df[\'formula\'].apply(lambda x:Composition(x))\nignore_errors=False\nX = featurizer.featurize_dataframe(df=test_df,col_id=\'pmg_comp\',ignore_errors=ignore_errors)\ntest_X_df = X.drop(columns=[\'id\',\'formula\',prop,\'pmg_comp\'])\ntest_y_df = X[prop]\n\nlgbm = lgb.LGBMRegressor(\n    # device="gpu",\n    n_estimators=1170,\n    learning_rate=0.15,\n    num_leaves=273,\n)\nlgbm.fit(train_X_df,train_y_df)\npred=lgbm.predict(test_X_df)\nprint (mean_absolute_error(test_y_df,pred))\n\nf=open(\'AI-SinglePropertyPrediction-magnetic_moment-mag2d_chem-test-mae.csv\',\'w\')\nf.write(\'id,target,prediction\\n\')\nfor i,j,k in zip(test_ids,test_y_df.values,pred):\n    line=str(i)+\',\'+str(j)+\',\'+str(k)+\'\\n\'\n    f.write(line)\nf.close()\ncmd = \'zip AI-SinglePropertyPrediction-magnetic_moment-mag2d_chem-test-mae.csv.zip AI-SinglePropertyPrediction-magnetic_moment-mag2d_chem-test-mae.csv\'\nos.system(cmd)\ncmd=\'rm AI-SinglePropertyPrediction-magnetic_moment-mag2d_chem-test-mae.csv\'\nos.system(cmd)\n')


# In[ ]:





# # 3. SuperCon

# In[1]:


get_ipython().run_cell_magic('time', '', '\nfrom jarvis.db.figshare import  get_request_data\nimport pandas as pd\nimport os\nimport lightgbm as lgb\nimport json,zipfile\nimport numpy as np\nfrom sklearn.metrics import mean_absolute_error\nfrom pymatgen.core.composition import Composition\nfrom sklearn.model_selection import train_test_split\nfrom matminer.featurizers.base import MultipleFeaturizer\nfrom matminer.featurizers.composition import (ElementProperty, \n                                              Stoichiometry, \n                                              ValenceOrbital, \n                                              IonProperty)\nfrom jarvis.core.composition import Composition as JComposition\n\ndat = get_request_data(js_tag="supercon_chem.json",url="https://figshare.com/ndownloader/files/40719260")\ndf = pd.DataFrame(dat)\n\npath = "../../benchmarks/AI/SinglePropertyPrediction/supercon_chem_Tc.json.zip"\njs_tag = "supercon_chem_Tc.json"\nid_data = json.loads(zipfile.ZipFile(path).read(js_tag))\ntrain_ids = np.array(list(id_data[\'train\'].keys()),dtype=\'int\')\ntest_ids = np.array(list(id_data[\'test\'].keys()),dtype=\'int\')\ntrain_df = df[df[\'id\'].isin(train_ids)]\ntest_df = df[df[\'id\'].isin(test_ids)]\ncompo_feat = [\n    (Stoichiometry()),\n    (ElementProperty.from_preset("magpie")),\n    (ValenceOrbital(props=[\'frac\'])),\n    (IonProperty(fast=True))\n    ]\nfeaturizer = MultipleFeaturizer(compo_feat)   \nprop = \'Tc\'\n# Without using JComposition results in error in pymatgen composition module\ntrain_df[\'pmg_comp\']=train_df[\'formula\'].apply(lambda x:Composition.from_dict(JComposition.from_string(x).to_dict()))\nignore_errors=False\nX = featurizer.featurize_dataframe(df=train_df,col_id=\'pmg_comp\',ignore_errors=ignore_errors)\ntrain_X_df = X.drop(columns=[\'id\',\'formula\',prop,\'pmg_comp\'])\ntrain_y_df = X[prop]\n\ntest_df[\'pmg_comp\']=test_df[\'formula\'].apply(lambda x:Composition.from_dict(JComposition.from_string(x).to_dict()))\nignore_errors=False\nX = featurizer.featurize_dataframe(df=test_df,col_id=\'pmg_comp\',ignore_errors=ignore_errors)\ntest_X_df = X.drop(columns=[\'id\',\'formula\',prop,\'pmg_comp\'])\ntest_y_df = X[prop]\n\nlgbm = lgb.LGBMRegressor(\n    # device="gpu",\n    n_estimators=1170,\n    learning_rate=0.15,\n    num_leaves=273,\n)\nlgbm.fit(train_X_df,train_y_df)\npred=lgbm.predict(test_X_df)\nprint (mean_absolute_error(test_y_df,pred))\n\nf=open(\'AI-SinglePropertyPrediction-Tc-supercon_chem-test-mae.csv\',\'w\')\nf.write(\'id,target,prediction\\n\')\nfor i,j,k in zip(test_ids,test_y_df.values,pred):\n    line=str(i)+\',\'+str(j)+\',\'+str(k)+\'\\n\'\n    f.write(line)\nf.close()\ncmd = \'zip AI-SinglePropertyPrediction-Tc-supercon_chem-test-mae.csv.zip AI-SinglePropertyPrediction-Tc-supercon_chem-test-mae.csv\'\nos.system(cmd)\ncmd=\'rm AI-SinglePropertyPrediction-Tc-supercon_chem-test-mae.csv\'\nos.system(cmd)\n')


# In[ ]:





# # Structural models: exfoliation energy

# In[1]:


get_ipython().run_cell_magic('time', '', 'from jarvis.db.figshare import  data\nimport json,zipfile\nimport numpy as np\nimport pandas as pd\nfrom jarvis.db.jsonutils import loadjson,dumpjson\nimport os\nfrom jarvis.core.atoms import Atoms\nimport lightgbm as lgb\nfrom sklearn.metrics import mean_absolute_error\n\ndataset = \'dft_3d\'\nprop = "exfoliation_energy"\ndat = data(dataset)\ndf = pd.DataFrame(dat)\npath = "../../benchmarks/AI/SinglePropertyPrediction/dft_3d_exfoliation_energy.json.zip"\njs_tag = "dft_3d_exfoliation_energy.json"\n\nid_data = json.loads(zipfile.ZipFile(path).read(js_tag))\ntrain_ids = np.array(list(id_data[\'train\'].keys()))\nval_ids = np.array(list(id_data[\'val\'].keys()))\ntest_ids = np.array(list(id_data[\'test\'].keys()))\ntrain_df = df[df[\'jid\'].isin(train_ids)]\ntrain_df[\'structure\']=train_df[\'atoms\'].apply(lambda x:(Atoms.from_dict(x)).pymatgen_converter() )\nval_df = (df[df[\'jid\'].isin(val_ids)])\nval_df[\'structure\']=val_df[\'atoms\'].apply(lambda x:(Atoms.from_dict(x)).pymatgen_converter() )\ntest_df = (df[df[\'jid\'].isin(test_ids)])\ntest_df[\'structure\']=test_df[\'atoms\'].apply(lambda x:(Atoms.from_dict(x)).pymatgen_converter() )\n\nimportant_features = [\n    "mean CN_VoronoiNN",\n    "mean ordering parameter shell 1",\n    "mean neighbor distance variation",\n    "avg_dev CN_VoronoiNN",\n    "mean local difference in NValence",\n    "MagpieData mean NpUnfilled",\n    "MagpieData mean NsUnfilled",\n    "minimum local difference in Number",\n    "MagpieData mode GSmagmom",\n    "minimum local difference in Column",\n    "MagpieData mode NfUnfilled",\n    "MagpieData mode GSbandgap",\n    "MagpieData maximum MeltingT",\n    "avg_dev local difference in NdValence",\n    "minimum local difference in NpUnfilled",\n    "MagpieData maximum CovalentRadius",\n    "MagpieData mode NValence",\n    "MagpieData range NdUnfilled",\n    "range local difference in NfValence",\n    "avg_dev local difference in CovalentRadius",\n    "minimum local difference in NdValence",\n    "MagpieData mean NUnfilled",\n    "MagpieData minimum AtomicWeight",\n    "MagpieData mode NdUnfilled",\n    "minimum local difference in NdUnfilled",\n    "MagpieData mean MeltingT",\n    "avg_dev local difference in NValence",\n    "minimum local difference in MeltingT",\n    "range local difference in NUnfilled",\n    "MagpieData minimum NValence",\n    "MagpieData minimum NsUnfilled",\n    "minimum local difference in NpValence",\n    "mean ordering parameter shell 3",\n    "MagpieData minimum GSvolume_pa",\n    "minimum local difference in GSvolume_pa",\n    "MagpieData maximum Column",\n    "frac d valence electrons",\n    "MagpieData mode NpUnfilled",\n    "avg_dev local difference in GSbandgap",\n    "MagpieData minimum NdValence",\n    "minimum local difference in CovalentRadius",\n    "MagpieData avg_dev Row",\n    "MagpieData minimum Electronegativity",\n    "0-norm",\n    "MagpieData maximum SpaceGroupNumber",\n    "MagpieData range Electronegativity",\n    "compound possible",\n    "range local difference in Column",\n    "MagpieData mode NsValence",\n    "MagpieData mode NfValence",\n    "minimum local difference in NsUnfilled",\n    "MagpieData mode NUnfilled",\n    "minimum neighbor distance variation",\n    "MagpieData mean MendeleevNumber",\n    "MagpieData avg_dev GSvolume_pa",\n    "minimum local difference in GSmagmom",\n    "minimum local difference in GSbandgap",\n    "frac s valence electrons",\n    "MagpieData minimum NfValence",\n    "MagpieData maximum Row",\n    "MagpieData minimum GSmagmom",\n    "MagpieData range NpUnfilled",\n    "range local difference in Row",\n    "avg_dev local difference in NsValence",\n    "MagpieData minimum GSbandgap",\n    "mean local difference in SpaceGroupNumber",\n    "MagpieData minimum NdUnfilled",\n    "MagpieData minimum NUnfilled",\n    "minimum local difference in NfUnfilled",\n    "minimum local difference in NfValence",\n    "MagpieData minimum NpUnfilled",\n    "MagpieData mode NsUnfilled",\n    "avg_dev local difference in MendeleevNumber",\n    "max relative bond length",\n    "avg_dev local difference in AtomicWeight",\n    "10-norm",\n    "avg_dev neighbor distance variation",\n    "minimum local difference in NUnfilled",\n    "MagpieData minimum NfUnfilled",\n    "MagpieData mode Column",\n    "MagpieData avg_dev MendeleevNumber",\n    "MagpieData mode SpaceGroupNumber",\n    "range local difference in NfUnfilled",\n    "MagpieData mode GSvolume_pa",\n    "min relative bond length",\n    "MagpieData maximum NdValence",\n    "maximum CN_VoronoiNN",\n    "avg_dev local difference in NpValence",\n    "MagpieData avg_dev GSmagmom",\n    "avg_dev local difference in NpUnfilled",\n]\ndef StructureFeaturizer(\n    df_in, col_id="structure",y_col="exfoliation_energy", ignore_errors=True, chunksize=30\n):\n    """\n    Featurize a dataframe using Matminter Structure featurizer\n\n    Parameters\n    ----------\n    df : Pandas.DataFrame\n        DataFrame with a column named "structure"\n\n    Returns\n    -------\n    A DataFrame containing 273 features (columns)\n\n    """\n    \n    # For featurization\n    from matminer.featurizers.base import MultipleFeaturizer\n    from matminer.featurizers.composition import (\n        ElementProperty,\n        Stoichiometry,\n        ValenceOrbital,\n        IonProperty,\n    )\n    from matminer.featurizers.structure import (\n        SiteStatsFingerprint,\n        StructuralHeterogeneity,\n        ChemicalOrdering,\n        StructureComposition,\n        MaximumPackingEfficiency,\n    )\n\n    if isinstance(df_in, pd.Series):\n        df = df_in.to_frame()\n    else:\n        df = df_in\n    y=df_in[y_col]\n    # df[col_id] = df[col_id].apply(to_unitcell)\n\n    # 128 structural feature\n    struc_feat = [\n        SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"),\n        SiteStatsFingerprint.from_preset(\n            "LocalPropertyDifference_ward-prb-2017"\n        ),\n        StructuralHeterogeneity(),\n        MaximumPackingEfficiency(),\n        ChemicalOrdering(),\n    ]\n    # 145 compositional features\n    compo_feat = [\n        StructureComposition(Stoichiometry()),\n        StructureComposition(ElementProperty.from_preset("magpie")),\n        StructureComposition(ValenceOrbital(props=["frac"])),\n        StructureComposition(IonProperty(fast=True)),\n    ]\n    featurizer = MultipleFeaturizer(struc_feat + compo_feat)\n    # Set the chunksize used for Pool.map parallelisation\n    featurizer.set_chunksize(chunksize=chunksize)\n    featurizer.fit(df[col_id])\n    X = featurizer.featurize_dataframe(\n        df=df, col_id=col_id, ignore_errors=ignore_errors\n    )\n    # check failed entries\n    print("Featurization completed.")\n    failed = np.any(pd.isnull(X.iloc[:, df.shape[1] :]), axis=1)\n    if np.sum(failed) > 0:\n        print(f"Number failed: {np.sum(failed)}/{len(failed)}")\n    return X[important_features],y, failed\n\n\nX_train,y_train,train_failed=StructureFeaturizer(df_in=train_df)\nX_test,y_test,test_failed=StructureFeaturizer(df_in=test_df)\nX_train_nan = X_train.fillna(0)\nX_test_nan = X_test.fillna(0)\nlgbm = lgb.LGBMRegressor(\n    # device="gpu",\n    n_estimators=1740,\n    learning_rate=0.040552334327414057,\n    num_leaves=291,\n    max_depth=16,\n    min_data_in_leaf=14,\n)\nlgbm.fit(np.array(X_train_nan.values,dtype=\'float\'), np.array(y_train.values,dtype=\'float\'))\npred = lgbm.predict(np.array(X_test_nan.values,dtype=\'float\'))\nprint(\'MAE\',mean_absolute_error(y_test,pred))\n')


# In[31]:


import os
dataset='dft_3d'
fname='AI-SinglePropertyPrediction-'+prop+'-'+dataset+'-test-mae.csv'
f=open(fname,'w')
f.write('id,target,prediction\n')
for i,j,k in zip(test_ids,y_test,pred):
    line=str(i)+','+str(j)+','+str(k)+'\n'
    f.write(line)
f.close()

# cmd = 'zip '+fname+'.zip '+fname
# os.system(cmd)
# cmd='rm '+fname
# os.system(cmd)


# In[26]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


#gpaw conda
from jarvis.db.figshare import data
import pandas as pd

d=data('dft_3d')
df = pd.DataFrame(d)
prop='formation_energy_peratom'
df_kv = df[df[prop]!='na']


# In[6]:


df.columns


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
ax = df_kv[prop].plot.hist(bins=12, alpha=0.5)


# In[4]:


from jarvis.core.composition import Composition
from jarvis.core.specie import Specie
import numpy as np
from jarvis.ai.descriptors.cfid import get_chem_only_descriptors
from jarvis.ai.descriptors.elemental import get_element_fraction_desc

def mean_absolute_deviation(data, axis=None):
    """Get Mean absolute deviation."""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

X=[]
Y=[]
IDs=[]
for ii,i in df_kv.iterrows():
    comp=i['formula']
    desc=get_element_fraction_desc(comp)
    val=i[prop]
    X.append(desc)
    Y.append(val)
    IDs.append(ii)
X = np.array(X)
Y = np.array(Y).reshape(-1, 1)
IDs = np.array(IDs)
mad = mean_absolute_deviation(Y)
print('MAD:',mad)  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1, test_size=0.1)
rf = RandomForestRegressor(n_estimators=90, max_depth = 4, n_jobs=-1, random_state=0,bootstrap=False)
rf.fit(X_train,y_train)
pred=rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(list(y_test), pred))
mae = mean_absolute_error(list(y_test), pred)
print('RMSE is %.3f' % rmse)
print('MAE is %.3f' % mae)
print('R2 score is: %.3f' % r2_score(list(y_test), pred))
print('MAD:MAE',mad/mae)


# In[ ]:





# In[ ]:





# In[ ]:


from jarvis.core.composition import Composition
from jarvis.core.specie import Specie
import numpy as np
from jarvis.ai.descriptors.cfid import get_chem_only_descriptors
from jarvis.ai.descriptors.elemental import get_element_fraction_desc

def mean_absolute_deviation(data, axis=None):
    """Get Mean absolute deviation."""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

X=[]
Y=[]
IDs=[]
for ii,i in df_kv.iterrows():
    comp=i['formula']
    desc=get_chem_only_descriptors(comp)
    val=i[prop]
    X.append(desc)
    Y.append(val)
    IDs.append(ii)
X = np.array(X)
Y = np.array(Y).reshape(-1, 1)
IDs = np.array(IDs)
mad = mean_absolute_deviation(Y)
print('MAD:',mad)  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1, test_size=0.1)
rf = RandomForestRegressor()#n_estimators=90, max_depth = 4, n_jobs=-1, random_state=0,bootstrap=False)
rf.fit(X_train,y_train)
pred=rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(list(y_test), pred))
mae = mean_absolute_error(list(y_test), pred)
print('RMSE is %.3f' % rmse)
print('MAE is %.3f' % mae)
print('R2 score is: %.3f' % r2_score(list(y_test), pred))
print('MAD:MAE',mad/mae)


# In[ ]:





# In[ ]:





# In[33]:


from jarvis.core.composition import Composition
from jarvis.core.specie import Specie
import numpy as np
from jarvis.ai.descriptors.cfid import get_chem_only_descriptors
from jarvis.ai.descriptors.elemental import get_element_fraction_desc

def mean_absolute_deviation(data, axis=None):
    """Get Mean absolute deviation."""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

X=[]
Y=[]
IDs=[]
for ii,i in df_kv.iterrows():
    comp=i['formula']
    density=i['density']
    spg=int(i['spg_number'])
    desc=get_element_fraction_desc(comp)
    desc = np.append(np.append(desc,density),spg)
    val=i[prop]
    X.append(desc)
    Y.append(val)
    IDs.append(ii)
X = np.array(X)
Y = np.array(Y).reshape(-1, 1)
IDs = np.array(IDs)
mad = mean_absolute_deviation(Y)
print('MAD:',mad)  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1, test_size=0.1)
rf = RandomForestRegressor(n_estimators=90, max_depth = 4, n_jobs=-1, random_state=0,bootstrap=False)
rf.fit(X_train,y_train)
pred=rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(list(y_test), pred))
mae = mean_absolute_error(list(y_test), pred)
print('RMSE is %.3f' % rmse)
print('MAE is %.3f' % mae)
print('R2 score is: %.3f' % r2_score(list(y_test), pred))
print('MAD:MAE',mad/mae)


# In[32]:


density


# In[9]:





# In[32]:


#https://github.com/mathsphy/paper-ml-robustness-material-property/blob/main/myfunc.py
def StructureFeaturizer(
        df_in,
        col_id='structure',
        ignore_errors=True,
        chunksize=30
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
    from matminer.featurizers.composition import (ElementProperty, 
                                                  Stoichiometry, 
                                                  ValenceOrbital, 
                                                  IonProperty)
    from matminer.featurizers.structure import (SiteStatsFingerprint, 
                                                StructuralHeterogeneity,
                                                ChemicalOrdering, 
                                                StructureComposition, 
                                                MaximumPackingEfficiency)
    
    
    if isinstance(df_in, pd.Series):
        df = df_in.to_frame()
    else:
        df = df_in
    #df[col_id] = df[col_id].apply(to_unitcell)
    
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
    featurizer = MultipleFeaturizer(struc_feat+compo_feat)    
    # Set the chunksize used for Pool.map parallelisation
    featurizer.set_chunksize(chunksize=chunksize)
    featurizer.fit(df[col_id])
    X = featurizer.featurize_dataframe(df=df,col_id=col_id,ignore_errors=ignore_errors)  
    # check failed entries    
    print('Featurization completed.')
    failed = np.any(pd.isnull(X.iloc[:,df.shape[1]:]), axis=1)
    if np.sum(failed) > 0:
        print(f'Number failed: {np.sum(failed)}/{len(failed)}')
    return X,failed


# In[33]:


from tqdm import tqdm
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
import pandas as pd

d=data('dft_3d_2021')
df = pd.DataFrame(d)
prop='formation_energy_peratom'
#prop='bulk_modulus_kv'
df_kv = df[df[prop]!='na']

tqdm.pandas()
df_kv['structure']=df_kv['atoms'].progress_apply(lambda x: ((Atoms.from_dict(x)).get_primitive_atoms).pymatgen_converter())


# In[34]:


X,failed=StructureFeaturizer(df_kv)


# In[35]:


Xn = X.dropna()


# In[36]:


Xn


# In[37]:


important_features = ['mean CN_VoronoiNN', 'mean ordering parameter shell 1', 'mean neighbor distance variation', 'avg_dev CN_VoronoiNN', 'mean local difference in NValence', 'MagpieData mean NpUnfilled', 'MagpieData mean NsUnfilled', 'minimum local difference in Number', 'MagpieData mode GSmagmom', 'minimum local difference in Column', 'MagpieData mode NfUnfilled', 'MagpieData mode GSbandgap', 'MagpieData maximum MeltingT', 'avg_dev local difference in NdValence', 'minimum local difference in NpUnfilled', 'MagpieData maximum CovalentRadius', 'MagpieData mode NValence', 'MagpieData range NdUnfilled', 'range local difference in NfValence', 'avg_dev local difference in CovalentRadius', 'minimum local difference in NdValence', 'MagpieData mean NUnfilled', 'MagpieData minimum AtomicWeight', 'MagpieData mode NdUnfilled', 'minimum local difference in NdUnfilled', 'MagpieData mean MeltingT', 'avg_dev local difference in NValence', 'minimum local difference in MeltingT', 'range local difference in NUnfilled', 'MagpieData minimum NValence', 'MagpieData minimum NsUnfilled', 'minimum local difference in NpValence', 'mean ordering parameter shell 3', 'MagpieData minimum GSvolume_pa', 'minimum local difference in GSvolume_pa', 'MagpieData maximum Column', 'frac d valence electrons', 'MagpieData mode NpUnfilled', 'avg_dev local difference in GSbandgap', 'MagpieData minimum NdValence', 'minimum local difference in CovalentRadius', 'MagpieData avg_dev Row', 'MagpieData minimum Electronegativity', '0-norm', 'MagpieData maximum SpaceGroupNumber', 'MagpieData range Electronegativity', 'compound possible', 'range local difference in Column', 'MagpieData mode NsValence', 'MagpieData mode NfValence', 'minimum local difference in NsUnfilled', 'MagpieData mode NUnfilled', 'minimum neighbor distance variation', 'MagpieData mean MendeleevNumber', 'MagpieData avg_dev GSvolume_pa', 'minimum local difference in GSmagmom', 'minimum local difference in GSbandgap', 'frac s valence electrons', 'MagpieData minimum NfValence', 'MagpieData maximum Row', 'MagpieData minimum GSmagmom', 'MagpieData range NpUnfilled', 'range local difference in Row', 'avg_dev local difference in NsValence', 'MagpieData minimum GSbandgap', 'mean local difference in SpaceGroupNumber', 'MagpieData minimum NdUnfilled', 'MagpieData minimum NUnfilled', 'minimum local difference in NfUnfilled', 'minimum local difference in NfValence', 'MagpieData minimum NpUnfilled', 'MagpieData mode NsUnfilled', 'avg_dev local difference in MendeleevNumber', 'max relative bond length', 'avg_dev local difference in AtomicWeight', '10-norm', 'avg_dev neighbor distance variation', 'minimum local difference in NUnfilled', 'MagpieData minimum NfUnfilled', 'MagpieData mode Column', 'MagpieData avg_dev MendeleevNumber', 'MagpieData mode SpaceGroupNumber', 'range local difference in NfUnfilled', 'MagpieData mode GSvolume_pa', 'min relative bond length', 'MagpieData maximum NdValence', 'maximum CN_VoronoiNN', 'avg_dev local difference in NpValence', 'MagpieData avg_dev GSmagmom', 'avg_dev local difference in NpUnfilled']
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
# imp_mean = SimpleImputer(strategy='mean')
# imp_mean.fit(X)

# X_IMP = pd.DataFrame(imp_mean.transform(X), index=X.index, 
#                       columns=X.columns)
X_train,X_test,y_train,y_test = train_test_split(X[important_features], X[prop])
import lightgbm as lgb
lgbm = lgb.LGBMRegressor()
lgbm.fit(X_train.values,y_train.values)


# In[38]:


y_pred=lgbm.predict(X_test.values)


# In[39]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_pred)


# In[41]:


#X.to_csv('X_dft_3d.csv')


# In[ ]:




