import pandas as pd
import time
from tb3py.main import predict_for_poscar
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
df = pd.read_csv('../vasp_optb88vdw/ES-SinglePropertyPrediction-bandgap-dft_3d-test-mae.csv.zip')
for i,ii in df.iterrows():
  jid=ii['id']
  atoms=Atoms.from_dict(get_jid_data(jid=jid,dataset="dft_3d")['atoms'])
  print(jid,atoms)
  posname='POSCAR-'+jid+'.vasp'
  atoms.write_poscar(posname)
  t1=time.time()
  info = predict_for_poscar(posname)
  print (info)
  t2=time.time()
  print('Time taken',t2-t1)
  name=jid.replace('-','_')+'_'+atoms.composition.reduced_formula
  fname='ES-SinglePropertyPrediction-bandgap_'+name+'-dft_3d-test-mae.csv'
  f=open(fname,'w')
  line='id,prediction\n'
  f.write(line)
  line=jid+','+str(info['indirectgap'])+'\n'
  f.write(line)  
  f.close()
