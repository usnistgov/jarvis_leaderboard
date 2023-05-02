from jarvis.db.jsonutils import loadjson
from jarvis.db.jsonutils import dumpjson
from jarvis.db.figshare import data
import numpy as np
import pandas as pd
from collections import defaultdict

d=data('alignn_ff_db')
ids=loadjson('ids_train_val_test.json')
id_train=ids['id_train']
id_val=ids['id_val']
id_test=ids['id_test']
df=pd.DataFrame(d)
train=defaultdict()
val=defaultdict()
test=defaultdict()

for i,j in (df[df['jid'].isin(id_test)][['jid','forces']]).iterrows():
   test[j['jid']]=';'.join(map(str,np.array(j['forces']).flatten()))

for i,j in (df[df['jid'].isin(id_train)][['jid','forces']]).iterrows():
   train[j['jid']]=';'.join(map(str,np.array(j['forces']).flatten()))

for i,j in (df[df['jid'].isin(id_val)][['jid','forces']]).iterrows():
   val[j['jid']]=';'.join(map(str,np.array(j['forces']).flatten()))

info=defaultdict()
info['train']=train
info['val']=val
info['test']=test
print ('test',len(test))
dumpjson(data=info,filename='alignn_ff_db_forces.json')
