

import pandas as pd
df= pd.read_csv('ssub.csv')
test_size = 0.2
n_train = int(len(df)*(1-test_size))
n_test = int(len(df)*test_size)

train_df = df[:n_train]
test_df=df[-n_test:]

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
lgbm.fit(X_train,y_train)
pred=lgbm.predict(X_test)
print (mean_absolute_error(y_test,pred))
f=open('SinglePropertyPrediction-test-formula_energy-ssub-AI-mae.csv','w')
f.write('id,target,prediction\n')
for i,j,k in zip(test_ids,y_test,pred):
    line=str(i)+','+str(j)+','+str(k)+'\n'
    f.write(line)
f.close()


