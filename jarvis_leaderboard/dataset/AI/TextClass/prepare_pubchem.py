import pandas as pd
from jarvis.db.jsonutils import dumpjson
#https://doi.org/10.6084/m9.figshare.22351717
df_csv = pd.read_csv('/wrk/knc6/AtomNLP/Summarize/pubchem.csv')
ratio=0.8
n_train = int(len(df_csv) * ratio)
n_test = len(df_csv) - n_train

traindf = df_csv[0:n_train]
testdf = df_csv[n_train:-1]


key='title'

fname='pubchem_categories.json'

value='label_name'
mem={}
train={}
test={}
for i,ii in traindf.iterrows():
    id=str(ii['id']) #.values
    label=str(ii['label_name'])
    train[id]=label
for i,ii in testdf.iterrows():
    id=str(ii['id']) #.values
    label=str(ii['label_name'])
    test[id]=label
mem['train']=train
mem['test']=test
dumpjson(filename=fname,data=mem)



    




