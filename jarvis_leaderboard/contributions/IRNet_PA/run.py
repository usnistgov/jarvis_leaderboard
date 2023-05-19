import pandas as pd
df= pd.read_csv('ssub.csv')

test_size = 0.2
val_size = 0.2
n_train = int(len(df)*(1-(val_size+test_size)))
n_val = int(len(df)*val_size)
n_test = int(len(df)*test_size)
#for leaderboard
train_df = df[:n_train]
val_df = df[n_train:n_train+n_val]
test_df = df[n_train+n_val:n_train+n_val+n_test]

import data_utils
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from pymatgen.core import Composition
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                          cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])

feature_labels = feature_calculators.feature_labels()

def formula_to_pa(df):
    
    df['comp_obj'] = df['formula'].apply(lambda x: Composition(x))
    df = feature_calculators.featurize_dataframe(df, col_id='comp_obj')
    df = df[~df[feature_labels].isnull().any(axis=1)]


    X_df = np.array(df[feature_labels],dtype='float')    
    y_df = np.array(df['formula_energy'],dtype='float')
    list_df = list(df['id'])

    return X_df, y_df, list_df

X_train, y_train, train_ids = formula_to_pa(train_df)
X_val, y_val, val_ids = formula_to_pa(val_df)
X_test, y_test, test_ids = formula_to_pa(test_df)


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
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Network from "Enabling deeper learning on big data for materials informatics applications"
#Link to paper: https://dl.acm.org/doi/10.1145/3292500.3330703 https://www.nature.com/articles/s41598-021-83193-1
in_layer = Input(shape=(X_train.shape[1],))

layer_1 = Dense(1024)(in_layer)
layer_1 = BatchNormalization()(layer_1)
layer_1 = Activation('relu')(layer_1)

fcc_1 = Dense(1024)(in_layer)
gsk_1 = add([fcc_1, layer_1])

layer_2 = Dense(1024)(gsk_1)
layer_2 = BatchNormalization()(layer_2)
layer_2 = Activation('relu')(layer_2)

gsk_2 = add([gsk_1, layer_2])

layer_3 = Dense(1024)(gsk_2)
layer_3 = BatchNormalization()(layer_3)
layer_3 = Activation('relu')(layer_3)

gsk_3 = add([gsk_2, layer_3])

layer_4 = Dense(1024)(gsk_3)
layer_4 = BatchNormalization()(layer_4)
layer_4 = Activation('relu')(layer_4)

gsk_4 = add([gsk_3, layer_4])

layer_5 = Dense(512)(gsk_4)
layer_5 = BatchNormalization()(layer_5)
layer_5 = Activation('relu')(layer_5)

mcc_5 = Dense(512)(gsk_4)
msk_5 = add([mcc_5, layer_5])

layer_6 = Dense(512)(msk_5)
layer_6 = BatchNormalization()(layer_6)
layer_6 = Activation('relu')(layer_6)

msk_6 = add([msk_5, layer_6])

layer_7 = Dense(512)(msk_6)
layer_7 = BatchNormalization()(layer_7)
layer_7 = Activation('relu')(layer_7)

msk_7 = add([msk_6, layer_7])

layer_8 = Dense(256)(msk_7)
layer_8 = BatchNormalization()(layer_8)
layer_8 = Activation('relu')(layer_8)

mcc_8 = Dense(256)(msk_7)
msk_8 = add([mcc_8, layer_8])

layer_9 = Dense(256)(msk_8)
layer_9 = BatchNormalization()(layer_9)
layer_9 = Activation('relu')(layer_9)

msk_9 = add([msk_8, layer_9])

layer_10 = Dense(256)(msk_9)
layer_10 = BatchNormalization()(layer_10)
layer_10 = Activation('relu')(layer_10)

msk_10 = add([msk_9, layer_10])

layer_11 = Dense(128)(layer_10)
layer_11 = BatchNormalization()(layer_11)
layer_11 = Activation('relu')(layer_11)

mcc_11 = Dense(128)(msk_10)
msk_11 = add([mcc_11, layer_11])

layer_12 = Dense(128)(msk_11)
layer_12 = BatchNormalization()(layer_12)
layer_12 = Activation('relu')(layer_12)

msk_12 = add([msk_11, layer_12])

layer_13 = Dense(128)(msk_12)
layer_13 = BatchNormalization()(layer_13)
layer_13 = Activation('relu')(layer_13)

msk_13 = add([msk_12, layer_13])

layer_14 = Dense(64)(msk_13)
layer_14 = BatchNormalization()(layer_14)
layer_14 = Activation('relu')(layer_14)

mcc_14 = Dense(64)(msk_13)
msk_14 = add([mcc_14, layer_14])

layer_15 = Dense(64)(msk_14)
layer_15 = BatchNormalization()(layer_15)
layer_15 = Activation('relu')(layer_15)

msk_15 = add([msk_14, layer_15])

layer_16 = Dense(32)(msk_15)
layer_16 = BatchNormalization()(layer_16)
layer_16 = Activation('relu')(layer_16)

mcc_16 = Dense(32)(msk_15)
msk_16 = add([mcc_16, layer_16])

out_layer = Dense(1)(msk_16)

model = Model(inputs=in_layer, outputs=out_layer)

adam = optimizers.Adam(lr=0.0001)
model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)

model.fit(X_train, y_train, verbose=2, validation_data=(X_val, y_val), epochs=3000, batch_size=32, callbacks=[es])

pred=model.predict(X_test)
print(mean_absolute_error(y_test,pred))

f=open('SinglePropertyPrediction-test-formula_energy-ssub-AI-mae.csv','w')
f.write('id,target,prediction\n')
for i,j,k in zip(test_ids,y_test,pred):
    line=str(i)+','+str(j)+','+str(k)+'\n'
    f.write(line)
f.close()