from jarvis.db.figshare import  get_request_data
import json,zipfile
import numpy as np
import pandas as pd
from jarvis.db.jsonutils import loadjson,dumpjson
import os
from jarvis.core.atoms import Atoms


take_val=100 #to save training time
dat = get_request_data(js_tag="ocp10k.json",url="https://figshare.com/ndownloader/files/40566122")
df = pd.DataFrame(dat)
path = "../../../benchmarks/AI/SinglePropertyPrediction/ocp10k_relaxed_energy.json.zip"
js_tag = "ocp10k_relaxed_energy.json"
id_data = json.loads(zipfile.ZipFile(path).read(js_tag))
train_ids = np.array(list(id_data['train'].keys()))
val_ids = np.array(list(id_data['val'].keys()))
test_ids = np.array(list(id_data['test'].keys()))
train_df = df[df['id'].isin(train_ids)]
val_df = (df[df['id'].isin(val_ids)])[:take_val]
test_df = (df[df['id'].isin(test_ids)])
print(train_df)
print(val_df)
print(test_df)
if not os.path.exists('config_example.json'):
 cmd = 'wget https://raw.githubusercontent.com/usnistgov/alignn/main/alignn/examples/sample_data/config_example.json -O config_example.json'
 os.system(cmd)
config = loadjson('config_example.json')
config['n_train'] = len(train_df)
config['n_val'] =  take_val #len(val_df)
config['n_test'] = len(test_df)
config['epochs'] = 100
config['batch_size'] = 32
dumpjson(data=config,filename='tmp_config.json')
f=open('DataDir/id_prop.csv','w')

for i,ii in train_df.iterrows():
    fname='DataDir/'+ii['id']
    #print(ii['atoms'])
    atoms = Atoms.from_dict(ii['atoms'])
    prop = ii['relaxed_energy']
    line=ii['id']+','+str(prop)+'\n'
    f.write(line)
    atoms.write_poscar(fname)
for i,ii in test_df.iterrows():
    fname='DataDir/'+ii['id']
    #print(ii['atoms'])
    atoms = Atoms.from_dict(ii['atoms'])
    prop = ii['relaxed_energy']
    line=ii['id']+','+str(prop)+'\n'
    f.write(line)
    atoms.write_poscar(fname)
for i,ii in test_df.iterrows():
    fname='DataDir/'+ii['id']
    #print(ii['atoms'])
    atoms = Atoms.from_dict(ii['atoms'])
    prop = ii['relaxed_energy']
    line=ii['id']+','+str(prop)+'\n'
    f.write(line)
    atoms.write_poscar(fname)
    #print (atoms)
    #print(prop)
    #break
f.close()

    
