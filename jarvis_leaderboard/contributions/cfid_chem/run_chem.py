#!/usr/bin/env python
# coding: utf-8

# # CFID-Chem: https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.2.083801

# # 1. SSUB

# In[1]:


#https://github.com/wolverton-research-group/qmpy/blob/master/qmpy/data/thermodata/ssub.dat
from jarvis.db.figshare import  get_request_data
import pandas as pd

dat = get_request_data(js_tag="ssub.json",url="https://figshare.com/ndownloader/files/40084921")
df = pd.DataFrame(dat)


# In[2]:


df


# In[3]:


import json,zipfile
import numpy as np
path = "../../benchmarks/AI/SinglePropertyPrediction/ssub_formula_energy.json.zip"
js_tag = "ssub_formula_energy.json"
id_data = json.loads(zipfile.ZipFile(path).read(js_tag))
train_ids = np.array(list(id_data['train'].keys()),dtype='int')
test_ids = np.array(list(id_data['test'].keys()),dtype='int')
print(len(train_ids),len(test_ids),len(train_ids)+len(test_ids))
train_df = df[df['id'].isin(train_ids)]
test_df = df[df['id'].isin(test_ids)]
# from jarvis.ai.descriptors.elemental import get_element_fraction_desc
# from jarvis.ai.descriptors.cfid import CFID,get_chem_only_descriptors
# from tqdm import tqdm
# import numpy as np
# from sklearn.model_selection import train_test_split
# import lightgbm as lgb
# from sklearn.metrics import mean_absolute_error

# lgbm = lgb.LGBMRegressor(
#     # device="gpu",
#     n_estimators=1170,
#     learning_rate=0.15,
#     num_leaves=273,
# )
# X_train=[]
# y_train=[]
# X_test=[]
# y_test=[]
# train_ids=[]
# test_ids=[]
# for i,ii in train_df.iterrows():
#     desc=get_element_fraction_desc(ii['formula'])
#     #desc=get_chem_only_descriptors(ii['composition'])[0]
#     X_train.append(desc)
#     y_train.append(ii['formula_energy'])
#     train_ids.append(ii['id'])
# for i,ii in test_df.iterrows():
#     desc=get_element_fraction_desc(ii['formula'])
#     #desc=get_chem_only_descriptors(ii['composition'])[0]
#     X_test.append(desc)
#     y_test.append(ii['formula_energy'])
#     test_ids.append(ii['id'])
    
# X_train=np.array(X_train,dtype='float')
# y_train=np.array(y_train,dtype='float')
# X_test=np.array(X_test,dtype='float')
# y_test=np.array(y_test,dtype='float')


# In[6]:


from jarvis.ai.descriptors.cfid import CFID,get_chem_only_descriptors
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

lgbm = lgb.LGBMRegressor(
    # device="gpu",
    n_estimators=1170,
    learning_rate=0.15,
    num_leaves=273,
)

X_train=[]
y_train=[]
X_test=[]
y_test=[]
train_ids=[]
test_ids=[]
for i,ii in train_df.iterrows():
    #desc=get_element_fraction_desc(ii['formula'])
    desc=get_chem_only_descriptors(ii['formula'])[0]
    X_train.append(desc)
    y_train.append(ii['formula_energy'])
    train_ids.append(ii['id'])

    
for i,ii in test_df.iterrows():
    #desc=get_element_fraction_desc(ii['formula'])
    desc=get_chem_only_descriptors(ii['formula'])[0]
    X_test.append(desc)
    y_test.append(ii['formula_energy'])
    test_ids.append(ii['id'])
    
X_train=np.array(X_train,dtype='float')
y_train=np.array(y_train,dtype='float')
X_test=np.array(X_test,dtype='float')
y_test=np.array(y_test,dtype='float')


# In[7]:


get_ipython().run_cell_magic('time', '', 'lgbm.fit(X_train,y_train)\npred=lgbm.predict(X_test)\nprint (mean_absolute_error(y_test,pred))\n')


# In[9]:


import os
f=open('AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv','w')
f.write('id,target,prediction\n')
for i,j,k in zip(test_ids,y_test,pred):
    line=str(i)+','+str(j)+','+str(k)+'\n'
    f.write(line)
f.close()
cmd = 'zip AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv.zip AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv'
os.system(cmd)
cmd='rm AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv'
os.system(cmd)


# In[10]:


from jarvis.db.figshare import  get_request_data
import pandas as pd
dat = get_request_data(js_tag="supercon_chem.json",url="https://figshare.com/ndownloader/files/40719260")
#http://supercon.nims.go.jp/index_en.html
#https://github.com/vstanev1/Supercon

df = pd.DataFrame(dat)


# In[11]:


df


# In[12]:


import json,zipfile
import numpy as np
path = "../../benchmarks/AI/SinglePropertyPrediction/supercon_chem_Tc.json.zip"
js_tag = "supercon_chem_Tc.json"
id_data = json.loads(zipfile.ZipFile(path).read(js_tag))
train_ids = np.array(list(id_data['train'].keys()),dtype='int')
test_ids = np.array(list(id_data['test'].keys()),dtype='int')
train_df = df[df['id'].isin(train_ids)]
test_df = df[df['id'].isin(test_ids)]


# In[13]:


from jarvis.ai.descriptors.cfid import CFID,get_chem_only_descriptors
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

lgbm = lgb.LGBMRegressor(
    # device="gpu",
    n_estimators=1170,
    learning_rate=0.15,
    num_leaves=273,
)

X_train=[]
y_train=[]
X_test=[]
y_test=[]
train_ids=[]
test_ids=[]
for i,ii in train_df.iterrows():
    #desc=get_element_fraction_desc(ii['formula'])
    desc=get_chem_only_descriptors(ii['formula'])[0]
    X_train.append(desc)
    y_train.append(ii['Tc'])
    train_ids.append(ii['id'])

    
for i,ii in test_df.iterrows():
    #desc=get_element_fraction_desc(ii['formula'])
    desc=get_chem_only_descriptors(ii['formula'])[0]
    X_test.append(desc)
    y_test.append(ii['Tc'])
    test_ids.append(ii['id'])
    
X_train=np.array(X_train,dtype='float')
y_train=np.array(y_train,dtype='float')
X_test=np.array(X_test,dtype='float')
y_test=np.array(y_test,dtype='float')


# In[14]:


get_ipython().run_cell_magic('time', '', 'lgbm.fit(X_train,y_train)\npred=lgbm.predict(X_test)\nprint (mean_absolute_error(y_test,pred))\n')


# In[15]:


import os
f=open('AI-SinglePropertyPrediction-Tc-supercon_chem-test-mae.csv','w')
f.write('id,target,prediction\n')
for i,j,k in zip(test_ids,y_test,pred):
    line=str(i)+','+str(j)+','+str(k)+'\n'
    f.write(line)
f.close()
cmd = 'zip AI-SinglePropertyPrediction-Tc-supercon_chem-test-mae.csv.zip AI-SinglePropertyPrediction-Tc-supercon_chem-test-mae.csv'
os.system(cmd)
cmd='rm AI-SinglePropertyPrediction-Tc-supercon_chem-test-mae.csv'
os.system(cmd)


# In[16]:


from sklearn.metrics import r2_score
r2_score(y_test,pred)


# In[ ]:





# In[17]:


import json,zipfile
import numpy as np
from jarvis.db.figshare import  get_request_data
import pandas as pd
dat = get_request_data(js_tag="mag2d_chem.json",url="https://figshare.com/ndownloader/files/40720004")
#http://supercon.nims.go.jp/index_en.html
#https://github.com/vstanev1/Supercon

df = pd.DataFrame(dat)
path = "../../benchmarks/AI/SinglePropertyPrediction/mag2d_chem_magnetic_moment.json.zip"
js_tag = "mag2d_chem_magnetic_moment.json"
id_data = json.loads(zipfile.ZipFile(path).read(js_tag))
train_ids = np.array(list(id_data['train'].keys()),dtype='int')
test_ids = np.array(list(id_data['test'].keys()),dtype='int')
train_df = df[df['id'].isin(train_ids)]
test_df = df[df['id'].isin(test_ids)]


# In[18]:


df


# In[19]:


from jarvis.ai.descriptors.cfid import CFID,get_chem_only_descriptors
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

lgbm = lgb.LGBMRegressor(
    # device="gpu",
    n_estimators=1170,
    learning_rate=0.15,
    num_leaves=273,
)

X_train=[]
y_train=[]
X_test=[]
y_test=[]
train_ids=[]
test_ids=[]
for i,ii in train_df.iterrows():
    #desc=get_element_fraction_desc(ii['formula'])
    desc=get_chem_only_descriptors(ii['formula'])[0]
    X_train.append(desc)
    y_train.append(ii['magnetic_moment'])
    train_ids.append(ii['id'])

    
for i,ii in test_df.iterrows():
    #desc=get_element_fraction_desc(ii['formula'])
    desc=get_chem_only_descriptors(ii['formula'])[0]
    X_test.append(desc)
    y_test.append(ii['magnetic_moment'])
    test_ids.append(ii['id'])
    
X_train=np.array(X_train,dtype='float')
y_train=np.array(y_train,dtype='float')
X_test=np.array(X_test,dtype='float')
y_test=np.array(y_test,dtype='float')


# In[20]:


get_ipython().run_cell_magic('time', '', 'lgbm.fit(X_train,y_train)\npred=lgbm.predict(X_test)\nprint (mean_absolute_error(y_test,pred))\n')


# In[21]:


import os
f=open('AI-SinglePropertyPrediction-magnetic_moment-mag2d_chem-test-mae.csv','w')
f.write('id,target,prediction\n')
for i,j,k in zip(test_ids,y_test,pred):
    line=str(i)+','+str(j)+','+str(k)+'\n'
    f.write(line)
f.close()
cmd = 'zip AI-SinglePropertyPrediction-magnetic_moment-mag2d_chem-test-mae.csv.zip AI-SinglePropertyPrediction-magnetic_moment-mag2d_chem-test-mae.csv'
os.system(cmd)
cmd='rm AI-SinglePropertyPrediction-magnetic_moment-mag2d_chem-test-mae.csv'
os.system(cmd)


# In[ ]:




