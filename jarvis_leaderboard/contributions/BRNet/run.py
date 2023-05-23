# %% [markdown]
# # BRNet training

# %% [markdown]
# # 1. SSUB

# %%
#https://github.com/wolverton-research-group/qmpy/blob/master/qmpy/data/thermodata/ssub.dat
from jarvis.db.figshare import  get_request_data
dat = get_request_data(js_tag="ssub.json",url="https://figshare.com/ndownloader/files/40084921")


# %%
import pandas as pd
df = pd.DataFrame(dat)

# %%
df

# %%
import json,zipfile
import numpy as np
path = "../../benchmarks/AI/SinglePropertyPrediction/ssub_formula_energy.json.zip"
js_tag = "ssub_formula_energy.json"
id_data = json.loads(zipfile.ZipFile(path).read(js_tag))

# %%
id_data.keys()

# %%
train_ids = np.array(list(id_data['train'].keys()),dtype='int')
test_ids = np.array(list(id_data['test'].keys()),dtype='int')

# %%
print(len(train_ids),len(test_ids),len(train_ids)+len(test_ids))

# %%
train_df = df[df['id'].isin(train_ids)]
test_df = df[df['id'].isin(test_ids)]

# %%
len(train_df),  len(test_df)

# %%
# %%time
from jarvis.ai.descriptors.elemental import get_element_fraction_desc
from jarvis.ai.descriptors.cfid import CFID,get_chem_only_descriptors
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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
    y_train.append(ii['formula_energy'])
    train_ids.append(ii['id'])
for i,ii in test_df.iterrows():
    desc=get_element_fraction_desc(ii['formula'])
    #desc=get_chem_only_descriptors(ii['composition'])[0]
    X_test.append(desc)
    y_test.append(ii['formula_energy'])
    test_ids.append(ii['id'])
    
X_train=np.array(X_train,dtype='float')
y_train=np.array(y_train,dtype='float')
X_test=np.array(X_test,dtype='float')
y_test=np.array(y_test,dtype='float')


# %%
#Split train into train-val split
n_train = int(len(X_train)*0.8)
n_val = len(X_train)-n_train
n_train,n_val,X_train.shape
X_train = X_train[:n_train]
y_train = y_train[:n_train]
X_val = X_train[-n_val:]
y_val = y_train[-n_val:]

# %%
X_train.shape,X_test.shape

# %%
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
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Activation
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %%
#Network from "BRNet: Branched Residual Network for Fast and Accurate Predictive Modeling of Materials Properties"
#Link to paper: https://epubs.siam.org/doi/10.1137/1.9781611977172.39
in_layer = Input(shape=(X_train.shape[1],))

layer_1 = Dense(1024)(in_layer)
layer_1 = LeakyReLU()(layer_1)

fcc_1 = Dense(1024)(in_layer)
gsk_1 = add([fcc_1, layer_1])

layer_2 = Dense(1024)(gsk_1)
layer_2 = LeakyReLU()(layer_2)

gsk_2 = add([gsk_1, layer_2])


rayer_1 = Dense(1024)(in_layer)
rayer_1 = LeakyReLU()(rayer_1)

rcc_1 = Dense(1024)(in_layer)
rsk_1 = add([rcc_1, rayer_1])

rayer_2 = Dense(1024)(rsk_1)
rayer_2 = LeakyReLU()(rayer_2)

rsk_2 = add([rsk_1, rayer_2])


mayer_1 = add([gsk_2, rsk_2])


layer_5 = Dense(512)(mayer_1)
layer_5 = LeakyReLU()(layer_5)

mcc_5 = Dense(512)(mayer_1)
msk_5 = add([mcc_5, layer_5])

layer_6 = Dense(512)(msk_5)
layer_6 = LeakyReLU()(layer_6)

msk_6 = add([msk_5, layer_6])

layer_7 = Dense(512)(msk_6)
layer_7 = LeakyReLU()(layer_7)

msk_7 = add([msk_6, layer_7])

layer_8 = Dense(256)(msk_7)
layer_8 = LeakyReLU()(layer_8)

mcc_8 = Dense(256)(msk_7)
msk_8 = add([mcc_8, layer_8])

layer_9 = Dense(256)(msk_8)
layer_9 = LeakyReLU()(layer_9)

msk_9 = add([msk_8, layer_9])

layer_10 = Dense(256)(msk_9)
layer_10 = LeakyReLU()(layer_10)

msk_10 = add([msk_9, layer_10])

layer_11 = Dense(128)(msk_10)
layer_11 = LeakyReLU()(layer_11)

mcc_11 = Dense(128)(msk_10)
msk_11 = add([mcc_11, layer_11])

layer_12 = Dense(128)(msk_11)
layer_12 = LeakyReLU()(layer_12)

msk_12 = add([msk_11, layer_12])

layer_13 = Dense(128)(msk_12)
layer_13 = LeakyReLU()(layer_13)

msk_13 = add([msk_12, layer_13])

layer_14 = Dense(64)(msk_13)
layer_14 = LeakyReLU()(layer_14)

mcc_14 = Dense(64)(msk_13)
msk_14 = add([mcc_14, layer_14])

layer_15 = Dense(64)(msk_14)
layer_15 = LeakyReLU()(layer_15)

msk_15 = add([msk_14, layer_15])

layer_16 = Dense(32)(msk_15)
layer_16 = LeakyReLU()(layer_16)

mcc_16 = Dense(32)(msk_15)
msk_16 = add([mcc_16, layer_16])

out_layer = Dense(1)(msk_16)

model = Model(inputs=in_layer, outputs=out_layer)

adam = optimizers.Adam(lr=0.0001)
model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)


# %%
# %%time
model.fit(X_train, y_train, verbose=2, validation_data=(X_val, y_val), epochs=3000, batch_size=32, callbacks=[es])

# %%
pred=model.predict(X_test)
print(mean_absolute_error(y_test,pred))

# %%
# pred=model.predict(X_test)
# print(mean_absolute_error(y_test,pred))
# 0.17889024475861343

# %%
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


# %% [markdown]
# # 2. SuperCon

# %%
from jarvis.db.figshare import  get_request_data
import pandas as pd
dat = get_request_data(js_tag="supercon_chem.json",url="https://figshare.com/ndownloader/files/40719260")
#http://supercon.nims.go.jp/index_en.html
#https://github.com/vstanev1/Supercon

df = pd.DataFrame(dat)

# %%
df

# %%
import json,zipfile
import numpy as np
path = "../../benchmarks/AI/SinglePropertyPrediction/supercon_chem_Tc.json.zip"
js_tag = "supercon_chem_Tc.json"
id_data = json.loads(zipfile.ZipFile(path).read(js_tag))
train_ids = np.array(list(id_data['train'].keys()),dtype='int')
test_ids = np.array(list(id_data['test'].keys()),dtype='int')
train_df = df[df['id'].isin(train_ids)]
test_df = df[df['id'].isin(test_ids)]

# %%
from jarvis.ai.descriptors.elemental import get_element_fraction_desc
from jarvis.ai.descriptors.cfid import CFID,get_chem_only_descriptors
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


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

# %%
X_train.shape,X_test.shape

# %%
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

#Network from "BRNet: Branched Residual Network for Fast and Accurate Predictive Modeling of Materials Properties"
#Link to paper: https://epubs.siam.org/doi/10.1137/1.9781611977172.39
in_layer = Input(shape=(X_train.shape[1],))

layer_1 = Dense(1024)(in_layer)
layer_1 = LeakyReLU()(layer_1)

fcc_1 = Dense(1024)(in_layer)
gsk_1 = add([fcc_1, layer_1])

layer_2 = Dense(1024)(gsk_1)
layer_2 = LeakyReLU()(layer_2)

gsk_2 = add([gsk_1, layer_2])


rayer_1 = Dense(1024)(in_layer)
rayer_1 = LeakyReLU()(rayer_1)

rcc_1 = Dense(1024)(in_layer)
rsk_1 = add([rcc_1, rayer_1])

rayer_2 = Dense(1024)(rsk_1)
rayer_2 = LeakyReLU()(rayer_2)

rsk_2 = add([rsk_1, rayer_2])


mayer_1 = add([gsk_2, rsk_2])


layer_5 = Dense(512)(mayer_1)
layer_5 = LeakyReLU()(layer_5)

mcc_5 = Dense(512)(mayer_1)
msk_5 = add([mcc_5, layer_5])

layer_6 = Dense(512)(msk_5)
layer_6 = LeakyReLU()(layer_6)

msk_6 = add([msk_5, layer_6])

layer_7 = Dense(512)(msk_6)
layer_7 = LeakyReLU()(layer_7)

msk_7 = add([msk_6, layer_7])

layer_8 = Dense(256)(msk_7)
layer_8 = LeakyReLU()(layer_8)

mcc_8 = Dense(256)(msk_7)
msk_8 = add([mcc_8, layer_8])

layer_9 = Dense(256)(msk_8)
layer_9 = LeakyReLU()(layer_9)

msk_9 = add([msk_8, layer_9])

layer_10 = Dense(256)(msk_9)
layer_10 = LeakyReLU()(layer_10)

msk_10 = add([msk_9, layer_10])

layer_11 = Dense(128)(msk_10)
layer_11 = LeakyReLU()(layer_11)

mcc_11 = Dense(128)(msk_10)
msk_11 = add([mcc_11, layer_11])

layer_12 = Dense(128)(msk_11)
layer_12 = LeakyReLU()(layer_12)

msk_12 = add([msk_11, layer_12])

layer_13 = Dense(128)(msk_12)
layer_13 = LeakyReLU()(layer_13)

msk_13 = add([msk_12, layer_13])

layer_14 = Dense(64)(msk_13)
layer_14 = LeakyReLU()(layer_14)

mcc_14 = Dense(64)(msk_13)
msk_14 = add([mcc_14, layer_14])

layer_15 = Dense(64)(msk_14)
layer_15 = LeakyReLU()(layer_15)

msk_15 = add([msk_14, layer_15])

layer_16 = Dense(32)(msk_15)
layer_16 = LeakyReLU()(layer_16)

mcc_16 = Dense(32)(msk_15)
msk_16 = add([mcc_16, layer_16])

out_layer = Dense(1)(msk_16)

model = Model(inputs=in_layer, outputs=out_layer)

adam = optimizers.Adam(lr=0.0001)
model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)


# %%
# %%time
model.fit(X_train, y_train, verbose=2, validation_data=(X_val, y_val), epochs=3000, batch_size=32, callbacks=[es])

# %%
pred=model.predict(X_test)
print(mean_absolute_error(y_test,pred))

# %%
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


# %% [markdown]
# # 3. Magnetic2DChem

# %%
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

# %%
df

# %%
from jarvis.ai.descriptors.elemental import get_element_fraction_desc
from jarvis.ai.descriptors.cfid import CFID,get_chem_only_descriptors
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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

# %%
X_train.shape,X_test.shape

# %%
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

#Network from "BRNet: Branched Residual Network for Fast and Accurate Predictive Modeling of Materials Properties"
#Link to paper: https://epubs.siam.org/doi/10.1137/1.9781611977172.39
in_layer = Input(shape=(X_train.shape[1],))

layer_1 = Dense(1024)(in_layer)
layer_1 = LeakyReLU()(layer_1)

fcc_1 = Dense(1024)(in_layer)
gsk_1 = add([fcc_1, layer_1])

layer_2 = Dense(1024)(gsk_1)
layer_2 = LeakyReLU()(layer_2)

gsk_2 = add([gsk_1, layer_2])


rayer_1 = Dense(1024)(in_layer)
rayer_1 = LeakyReLU()(rayer_1)

rcc_1 = Dense(1024)(in_layer)
rsk_1 = add([rcc_1, rayer_1])

rayer_2 = Dense(1024)(rsk_1)
rayer_2 = LeakyReLU()(rayer_2)

rsk_2 = add([rsk_1, rayer_2])


mayer_1 = add([gsk_2, rsk_2])


layer_5 = Dense(512)(mayer_1)
layer_5 = LeakyReLU()(layer_5)

mcc_5 = Dense(512)(mayer_1)
msk_5 = add([mcc_5, layer_5])

layer_6 = Dense(512)(msk_5)
layer_6 = LeakyReLU()(layer_6)

msk_6 = add([msk_5, layer_6])

layer_7 = Dense(512)(msk_6)
layer_7 = LeakyReLU()(layer_7)

msk_7 = add([msk_6, layer_7])

layer_8 = Dense(256)(msk_7)
layer_8 = LeakyReLU()(layer_8)

mcc_8 = Dense(256)(msk_7)
msk_8 = add([mcc_8, layer_8])

layer_9 = Dense(256)(msk_8)
layer_9 = LeakyReLU()(layer_9)

msk_9 = add([msk_8, layer_9])

layer_10 = Dense(256)(msk_9)
layer_10 = LeakyReLU()(layer_10)

msk_10 = add([msk_9, layer_10])

layer_11 = Dense(128)(msk_10)
layer_11 = LeakyReLU()(layer_11)

mcc_11 = Dense(128)(msk_10)
msk_11 = add([mcc_11, layer_11])

layer_12 = Dense(128)(msk_11)
layer_12 = LeakyReLU()(layer_12)

msk_12 = add([msk_11, layer_12])

layer_13 = Dense(128)(msk_12)
layer_13 = LeakyReLU()(layer_13)

msk_13 = add([msk_12, layer_13])

layer_14 = Dense(64)(msk_13)
layer_14 = LeakyReLU()(layer_14)

mcc_14 = Dense(64)(msk_13)
msk_14 = add([mcc_14, layer_14])

layer_15 = Dense(64)(msk_14)
layer_15 = LeakyReLU()(layer_15)

msk_15 = add([msk_14, layer_15])

layer_16 = Dense(32)(msk_15)
layer_16 = LeakyReLU()(layer_16)

mcc_16 = Dense(32)(msk_15)
msk_16 = add([mcc_16, layer_16])

out_layer = Dense(1)(msk_16)

model = Model(inputs=in_layer, outputs=out_layer)

adam = optimizers.Adam(lr=0.0001)
model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)


# %%
# %%time
model.fit(X_train, y_train, verbose=2, validation_data=(X_val, y_val), epochs=3000, batch_size=32, callbacks=[es])

# %%
pred=model.predict(X_test)
print(mean_absolute_error(y_test,pred))

# %%
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


# %%



