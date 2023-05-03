#https://github.com/wolverton-research-group/qmpy/blob/master/qmpy/data/thermodata/ssub.dat
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



df=pd.read_csv('ssub.dat',sep=' ',header=None)
df.columns=['formula','formula_energy']
df=df.sort_values('formula_energy', ascending=True).drop_duplicates('formula').sort_index()
df=df.sample(frac=1)
df['id']=df.index
df.to_csv('ssub.csv')
df = pd.read_csv('ssub.csv')
# 145 compositional features
compo_feat = [
    (Stoichiometry()),
    (ElementProperty.from_preset("magpie")),
    (ValenceOrbital(props=['frac'])),
    (IonProperty(fast=True))
    ]
featurizer = MultipleFeaturizer(compo_feat)   
df['pmg_comp']=df['formula'].apply(lambda x:Composition(x))
ignore_errors=False
X = featurizer.featurize_dataframe(df=df,col_id='pmg_comp',ignore_errors=ignore_errors)

X_df = X.drop(columns=['Unnamed: 0','formula','formula_energy','pmg_comp'])
Y_df=X['formula_energy']
ids=df['id']

test_size = 0.2
n_train = int(len(df)*(1-test_size))
n_test = int(len(df)*test_size)


X_train = X_df[0:n_train]
Y_train = Y_df[0:n_train]


X_test = X_df[-n_test:]
Y_test = Y_df[-n_test:]

test_ids=ids[-n_test:]
#X_train, X_test, y_train, y_test = train_test_split(X_df, Y_df, test_size=0.2, random_state=42,shuffle=False)

print ('X_train',X_train)
lgbm = lgb.LGBMRegressor(
    # device="gpu",
    n_estimators=1170,
    learning_rate=0.15,
    num_leaves=273,
)
lgbm.fit(X_train.drop(columns=['id']),Y_train)
pred=lgbm.predict(X_test.drop(columns=['id']))
print (mean_absolute_error(Y_test,pred))


f=open('SinglePropertyPrediction-test-formula_energy-ssub-AI-mae.csv','w')
f.write('id,target,prediction\n')
for i,j,k in zip(test_ids,Y_test,pred):
    line=str(i)+','+str(j)+','+str(k)+'\n'
    f.write(line)
f.close()



