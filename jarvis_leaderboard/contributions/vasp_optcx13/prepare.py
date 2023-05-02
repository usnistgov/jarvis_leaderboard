import os
import pandas as pd
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
df = pd.read_csv('SinglePropertyPrediction-test-bulk_modulus-dft_3d-ES-mae.csv.zip')
for ii,i in df.iterrows():
    jid=i['id']
    prediction=i['prediction']
    #print (jid,prediction)
    dat=get_jid_data(jid=jid,dataset='dft_3d')['formula']
    name=jid.replace('-','_')+'_'+dat
    fname = 'SinglePropertyPrediction-test-bulk_modulus_'+name+'-dft_3d-ES-mae.csv' 
    f=open(fname,'w')
    line='id,prediction\n'
    f.write(line)
    line=jid+','+str(prediction)
    f.write(line)
    f.close()
    cmd = 'zip '+fname+'.zip '+fname
    os.system(cmd)
    cmd = 'rm '+fname
    os.system(cmd)
    #print (fname)
    #print()

