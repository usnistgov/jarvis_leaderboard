#!/usr/bin/env python
# coding: utf-8

# # ElemNet training

# # 1. SSUB

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


# In[7]:


print(len(train_ids),len(test_ids),len(train_ids)+len(test_ids))


# In[8]:


train_df = df[df['id'].isin(train_ids)]
test_df = df[df['id'].isin(test_ids)]


# In[9]:


len(train_df),  len(test_df)


# In[10]:


get_ipython().run_cell_magic('time', '', 'from jarvis.ai.descriptors.elemental import get_element_fraction_desc\nfrom jarvis.ai.descriptors.cfid import CFID,get_chem_only_descriptors\nfrom tqdm import tqdm\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nimport lightgbm as lgb\nfrom sklearn.metrics import mean_absolute_error\n\nlgbm = lgb.LGBMRegressor(\n    # device="gpu",\n    n_estimators=1170,\n    learning_rate=0.15,\n    num_leaves=273,\n)\nX_train=[]\ny_train=[]\nX_test=[]\ny_test=[]\ntrain_ids=[]\ntest_ids=[]\nfor i,ii in train_df.iterrows():\n    desc=get_element_fraction_desc(ii[\'formula\'])\n    #desc=get_chem_only_descriptors(ii[\'composition\'])[0]\n    X_train.append(desc)\n    y_train.append(ii[\'formula_energy\'])\n    train_ids.append(ii[\'id\'])\nfor i,ii in test_df.iterrows():\n    desc=get_element_fraction_desc(ii[\'formula\'])\n    #desc=get_chem_only_descriptors(ii[\'composition\'])[0]\n    X_test.append(desc)\n    y_test.append(ii[\'formula_energy\'])\n    test_ids.append(ii[\'id\'])\n    \nX_train=np.array(X_train,dtype=\'float\')\ny_train=np.array(y_train,dtype=\'float\')\nX_test=np.array(X_test,dtype=\'float\')\ny_test=np.array(y_test,dtype=\'float\')\n')


# In[11]:


#Split train into train-val split
n_train = int(len(X_train)*0.8)
n_val = len(X_train)-n_train
n_train,n_val,X_train.shape
X_train = X_train[:n_train]
y_train = y_train[:n_train]
X_val = X_train[-n_val:]
y_val = y_train[-n_val:]


# In[12]:


X_train.shape,X_test.shape


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


# In[15]:


get_ipython().run_cell_magic('time', '', 'model.fit(X_train, y_train, verbose=2, validation_data=(X_val, y_val), epochs=3000, batch_size=32, callbacks=[es])\n')


# In[16]:


pred=model.predict(X_test)
print(mean_absolute_error(y_test,pred))


# In[17]:


# pred=model.predict(X_test)
# print(mean_absolute_error(y_test,pred))
# 0.17889024475861343


# In[18]:


import os
f=open('AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv','w')
f.write('id,target,prediction\n')
for i,j,k in zip(test_ids,y_test,pred[0]):
    line=str(i)+','+str(j)+','+str(k)+'\n'
    f.write(line)
f.close()
cmd = 'zip AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv.zip AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv'
os.system(cmd)
cmd='rm AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv'
os.system(cmd)


# # 2. SuperCon

# In[19]:


from jarvis.db.figshare import  get_request_data
import pandas as pd
dat = get_request_data(js_tag="supercon_chem.json",url="https://figshare.com/ndownloader/files/40719260")
#http://supercon.nims.go.jp/index_en.html
#https://github.com/vstanev1/Supercon

df = pd.DataFrame(dat)


# In[20]:


df


# In[21]:


import json,zipfile
import numpy as np
path = "../../benchmarks/AI/SinglePropertyPrediction/supercon_chem_Tc.json.zip"
js_tag = "supercon_chem_Tc.json"
id_data = json.loads(zipfile.ZipFile(path).read(js_tag))
train_ids = np.array(list(id_data['train'].keys()),dtype='int')
test_ids = np.array(list(id_data['test'].keys()),dtype='int')
train_df = df[df['id'].isin(train_ids)]
test_df = df[df['id'].isin(test_ids)]


# In[22]:


from jarvis.ai.descriptors.elemental import get_element_fraction_desc
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
    desc=get_element_fraction_desc(ii['formula'])
    #desc=get_chem_only_descriptors(ii['composition'])[0]
    X_train.append(desc)
    y_train.append(ii['Tc'])
    train_ids.append(ii['id'])
for i,ii in test_df.iterrows():
    desc=get_element_fraction_desc(ii['formula'])
    #desc=get_chem_only_descriptors(ii['composition'])[0]
    X_test.append(desc)
    y_test.append(ii['Tc'])
    test_ids.append(ii['id'])
    
X_train=np.array(X_train,dtype='float')
y_train=np.array(y_train,dtype='float')
X_test=np.array(X_test,dtype='float')
y_test=np.array(y_test,dtype='float')
#Split train into train-val split
n_train = int(len(X_train)*0.8)
n_val = len(X_train)-n_train
n_train,n_val,X_train.shape
X_train = X_train[:n_train]
y_train = y_train[:n_train]
X_val = X_train[-n_val:]
y_val = y_train[-n_val:]


# In[23]:


X_train.shape,X_test.shape


# In[24]:


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


# In[25]:


get_ipython().run_cell_magic('time', '', 'model.fit(X_train, y_train, verbose=2, validation_data=(X_val, y_val), epochs=3000, batch_size=32, callbacks=[es])\n')


# In[26]:


pred=model.predict(X_test)
print(mean_absolute_error(y_test,pred))


# In[27]:


import os
f=open('AI-SinglePropertyPrediction-Tc-supercon_chem-test-mae.csv','w')
f.write('id,target,prediction\n')
for i,j,k in zip(test_ids,y_test,pred):
    line=str(i)+','+str(j)+','+str(k[0])+'\n'
    f.write(line)
f.close()
cmd = 'zip AI-SinglePropertyPrediction-Tc-supercon_chem-test-mae.csv.zip AI-SinglePropertyPrediction-Tc-supercon_chem-test-mae.csv'
os.system(cmd)
cmd='rm AI-SinglePropertyPrediction-Tc-supercon_chem-test-mae.csv'
os.system(cmd)


# # Magnetic2DChem

# In[30]:


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


# In[31]:


df


# In[32]:


from jarvis.ai.descriptors.elemental import get_element_fraction_desc
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
    desc=get_element_fraction_desc(ii['formula'])
    #desc=get_chem_only_descriptors(ii['composition'])[0]
    X_train.append(desc)
    y_train.append(ii['magnetic_moment'])
    train_ids.append(ii['id'])
for i,ii in test_df.iterrows():
    desc=get_element_fraction_desc(ii['formula'])
    #desc=get_chem_only_descriptors(ii['composition'])[0]
    X_test.append(desc)
    y_test.append(ii['magnetic_moment'])
    test_ids.append(ii['id'])
    
X_train=np.array(X_train,dtype='float')
y_train=np.array(y_train,dtype='float')
X_test=np.array(X_test,dtype='float')
y_test=np.array(y_test,dtype='float')
#Split train into train-val split
n_train = int(len(X_train)*0.8)
n_val = len(X_train)-n_train
n_train,n_val,X_train.shape
X_train = X_train[:n_train]
y_train = y_train[:n_train]
X_val = X_train[-n_val:]
y_val = y_train[-n_val:]


# In[33]:


X_train.shape,X_test.shape


# In[34]:


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


# In[35]:


get_ipython().run_cell_magic('time', '', 'model.fit(X_train, y_train, verbose=2, validation_data=(X_val, y_val), epochs=3000, batch_size=32, callbacks=[es])\n')


# In[ ]:


pred=model.predict(X_test)
print(mean_absolute_error(y_test,pred))


# In[36]:


import os
f=open('AI-SinglePropertyPrediction-magnetic_moment-mag2d_chem-test-mae.csv','w')
f.write('id,target,prediction\n')
for i,j,k in zip(test_ids,y_test,pred):
    line=str(i)+','+str(j)+','+str(k[0])+'\n'
    f.write(line)
f.close()
cmd = 'zip AI-SinglePropertyPrediction-magnetic_moment-mag2d_chem-test-mae.csv.zip AI-SinglePropertyPrediction-magnetic_moment-mag2d_chem-test-mae.csv'
os.system(cmd)
cmd='rm AI-SinglePropertyPrediction-magnetic_moment-mag2d_chem-test-mae.csv'
os.system(cmd)


# In[ ]:




