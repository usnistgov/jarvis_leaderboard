#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://github.com/wolverton-research-group/qmpy/blob/master/qmpy/data/thermodata/ssub.dat
from jarvis.db.figshare import  get_request_data
dat = get_request_data(js_tag="ssub.json",url="https://figshare.com/ndownloader/files/40084921")


# In[2]:


import pandas as pd
df = pd.DataFrame(dat)


# In[3]:


df


# In[4]:


import json,zipfile
import numpy as np
path = "../../benchmarks/AI/SinglePropertyPrediction/ssub_formula_energy.json.zip"
js_tag = "ssub_formula_energy.json"
id_data = json.loads(zipfile.ZipFile(path).read(js_tag))


# In[5]:


id_data.keys()


# In[6]:


train_ids = np.array(list(id_data['train'].keys()),dtype='int')
test_ids = np.array(list(id_data['test'].keys()),dtype='int')


# In[9]:


print(len(train_ids),len(test_ids),len(train_ids)+len(test_ids))


# In[10]:


train_df = df[df['id'].isin(train_ids)]
test_df = df[df['id'].isin(test_ids)]


# In[10]:


len(train_df), len(val_df), len(test_df)


# In[12]:


get_ipython().run_cell_magic('time', '', 'from jarvis.ai.descriptors.elemental import get_element_fraction_desc\nfrom jarvis.ai.descriptors.cfid import CFID,get_chem_only_descriptors\nfrom tqdm import tqdm\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nimport lightgbm as lgb\nfrom sklearn.metrics import mean_absolute_error\n\nlgbm = lgb.LGBMRegressor(\n    # device="gpu",\n    n_estimators=1170,\n    learning_rate=0.15,\n    num_leaves=273,\n)\nX_train=[]\ny_train=[]\nX_test=[]\ny_test=[]\ntrain_ids=[]\ntest_ids=[]\nfor i,ii in train_df.iterrows():\n    desc=get_element_fraction_desc(ii[\'formula\'])\n    #desc=get_chem_only_descriptors(ii[\'composition\'])[0]\n    X_train.append(desc)\n    y_train.append(ii[\'formula_energy\'])\n    train_ids.append(ii[\'id\'])\nfor i,ii in test_df.iterrows():\n    desc=get_element_fraction_desc(ii[\'formula\'])\n    #desc=get_chem_only_descriptors(ii[\'composition\'])[0]\n    X_test.append(desc)\n    y_test.append(ii[\'formula_energy\'])\n    test_ids.append(ii[\'id\'])\n    \nX_train=np.array(X_train,dtype=\'float\')\ny_train=np.array(y_train,dtype=\'float\')\nX_test=np.array(X_test,dtype=\'float\')\ny_test=np.array(y_test,dtype=\'float\')\nlgbm.fit(X_train,y_train)\npred=lgbm.predict(X_test)\nprint (mean_absolute_error(y_test,pred))\n')


# In[19]:


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


# In[11]:


import data_utils
from jarvis.ai.descriptors.elemental import get_element_fraction_desc
from jarvis.ai.descriptors.cfid import CFID,get_chem_only_descriptors
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

elements = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K',
 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
 'Hg', 'Tl', 'Pb', 'Bi', 'Ac','Th', 'Pa', 'U', 'Np', 'Pu']

def formula_to_ef(df):
    df['comp_dict'] = df['formula'].apply(lambda x: data_utils.parse_formula(x))
    df['comp_fractions'] = df['comp_dict'].apply(lambda x: data_utils.get_fractions(x))
    df = df[~df['comp_fractions'].isnull()]

    for i,e in enumerate(elements):
        df[e] = [ x[i] for x in df['comp_fractions']]

    X_df = np.array(df[elements],dtype='float')    
    y_df = np.array(df['formula_energy'],dtype='float')
    list_df = list(df['id'])

    return X_df, y_df, list_df


# In[12]:


X_train, y_train, train_ids = formula_to_ef(train_df)
X_val, y_val, val_ids = formula_to_ef(val_df)
X_test, y_test, test_ids = formula_to_ef(test_df)


# In[13]:


import numpy as np
np.random.seed(1234567)
import tensorflow as tf
tf.random.set_seed(1234567)
import random
random.seed(1234567)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[14]:


#Network from "ElemNet: Deep Learning the Chemistry of Materials From Only Elemental Composition"
#Link to paper: https://www.nature.com/articles/s41598-018-35934-y
in_layer = Input(shape=(X_train.shape[1],))

layer_1 = Dense(1024)(in_layer)
layer_1 = Activation('relu')(layer_1)

layer_2 = Dense(1024)(layer_1)
layer_2 = Activation('relu')(layer_2)

layer_3 = Dense(1024)(layer_2)
layer_3 = Activation('relu')(layer_3)

layer_4 = Dense(1024)(layer_3)
layer_4 = Activation('relu')(layer_4)
layer_4 = Dropout(0.2)(layer_4, training=True)

layer_5 = Dense(512)(layer_4)
layer_5 = Activation('relu')(layer_5)

layer_6 = Dense(512)(layer_5)
layer_6 = Activation('relu')(layer_6)

layer_7 = Dense(512)(layer_6)
layer_7 = Activation('relu')(layer_7)
layer_7 = Dropout(0.1)(layer_7, training=True)

layer_8 = Dense(256)(layer_7)
layer_8 = Activation('relu')(layer_8)

layer_9 = Dense(256)(layer_8)
layer_9 = Activation('relu')(layer_9)

layer_10 = Dense(256)(layer_9)
layer_10 = Activation('relu')(layer_10)
layer_10 = Dropout(0.3)(layer_10, training=True)

layer_11 = Dense(128)(layer_10)
layer_11 = Activation('relu')(layer_11)

layer_12 = Dense(128)(layer_11)
layer_12 = Activation('relu')(layer_12)

layer_13 = Dense(128)(layer_12)
layer_13 = Activation('relu')(layer_13)
layer_13 = Dropout(0.2)(layer_13, training=True)

layer_14 = Dense(64)(layer_13)
layer_14 = Activation('relu')(layer_14)

layer_15 = Dense(64)(layer_14)
layer_15 = Activation('relu')(layer_15)

layer_16 = Dense(32)(layer_15)
layer_16 = Activation('relu')(layer_16)

out_layer = Dense(1)(layer_16)

model = Model(inputs=in_layer, outputs=out_layer)

adam = optimizers.Adam(lr=0.0001)
model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)


# In[16]:


get_ipython().run_cell_magic('time', '', 'model.fit(X_train, y_train, verbose=2, validation_data=(X_val, y_val), epochs=3000, batch_size=32, callbacks=[es])\n')


# In[17]:


pred=model.predict(X_test)
print(mean_absolute_error(y_test,pred))


# In[18]:


#f=open('SinglePropertyPrediction-test-formula_energy-ssub-AI-mae.csv','w')
f=open('AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv','w')
f.write('id,target,prediction\n')
for i,j,k in zip(test_ids,y_test,pred):
    line=str(i)+','+str(j)+','+str(k)+'\n'
    f.write(line)
f.close()


# In[19]:


import os
cmd = 'zip AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv.zip AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv'
os.system(cmd)


# In[ ]:




