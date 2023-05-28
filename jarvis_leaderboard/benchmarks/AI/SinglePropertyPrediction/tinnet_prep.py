#wget https://github.com/hlxin/tinnet/raw/master/data/tinnet_N.db -O tinnet_N.db
from jarvis.core.atoms import Atoms
import os
from ase.db import connect
import numpy as np
# main target
#adsorption energy
db = connect('tinnet_N.db')
ead = np.array([r['ead'] for r in db.select()])
images = [r.toatoms() for r in db.select()]
def ase_to_atoms(ase_atoms="", cartesian=True):
    """Convert ase structure to Atoms."""
    return Atoms(
        lattice_mat=ase_atoms.get_cell(),
        elements=ase_atoms.get_chemical_symbols(),
        coords=ase_atoms.get_positions(),
        cartesian=cartesian,
    )


cwd=os.getcwd()
if not os.path.exists('DataDir_tinnet_N'):
    os.makedirs('DataDir_tinnet_N')
os.chdir('DataDir_tinnet_N')

f=open('id_prop.csv','w')
mem=[]
count=0
for i,j in zip(images,ead):
    count+=1
    atoms = ase_to_atoms(i)
    fname='tinnet-'+str(count)
    atoms.write_poscar(fname)
    line=fname+','+str(j)+'\n'
    f.write(line)
    info={}
    info['id']='tinnet-'+str(count)
    info['atoms']=atoms.to_dict()
    info['ead']=j
    mem.append(info)
f.close()
os.chdir(cwd)
from jarvis.db.jsonutils import dumpjson
dumpjson(data=mem,filename='tinnet_N.json')
