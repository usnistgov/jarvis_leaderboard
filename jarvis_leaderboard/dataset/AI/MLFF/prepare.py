from jarvis.db.jsonutils import loadjson
from jarvis.db.figshare import data
d=data('alignn_ff_db')
ids=loadjson('ids_train_val_test.json')
id_train=ids['id_train']
id_val=ids['id_val']
id_test=ids['id_test']
import pandas as pd
df=pd.DataFrame(d)
from collections import defaultdict
train=defaultdict()
val=defaultdict()
test=defaultdict()
for i,j in (df[df['jid'].isin(id_test)][['jid','total_energy']]).iterrows():
   test[j['jid']]=j['total_energy']
for i,j in (df[df['jid'].isin(id_train)][['jid','total_energy']]).iterrows():
   train[j['jid']]=j['total_energy']
for i,j in (df[df['jid'].isin(id_val)][['jid','total_energy']]).iterrows():
   val[j['jid']]=j['total_energy']
info=defaultdict()
info['train']=train
info['val']=val
info['test']=test
from jarvis.db.jsonutils import dumpjson
dumpjson(data=info,filename='alignn_ff_db_total_energy.json')
