from jarvis.core.atoms import Atoms
import pandas as pd
from jarvis.db.jsonutils import dumpjson
df=pd.read_csv('id_prop.csv',header=None)
mem=[]
for i,ii in df.iterrows():
   mid=ii[0]
   #afer unzipping poscars.zip
   atoms=Atoms.from_poscar(mid)
   prop=ii[1]
   info={}
   info['id']=mid
   info['atoms']=atoms.to_dict()
   info['formation_energy']=prop
   mem.append(info)
dumpjson(data=mem,filename='formation_energy_mxene275.json')
   
