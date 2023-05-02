import zipfile
import json
import glob
import pandas as pd
import numpy as np
from jarvis.core.atoms import Atoms
import os
from jarvis.db.figshare import data
from alignn.ff.ff import AlignnAtomwiseCalculator,default_path

model_path = default_path()
calculator = AlignnAtomwiseCalculator(path=model_path)
#from m3gnet.models import M3GNet, M3GNetCalculator, Potential

#AI-MLFF-energy-alignn_ff_db-test-mae.csv.zip

#potential = Potential(M3GNet.load())
#calculator = M3GNetCalculator(potential=potential, stress_weight=0.01)
##wget https://figshare.com/ndownloader/files/40357663 -O mlearn.json.zip
 
def get_alignn_forces(atoms,rescale_factor=2.5):
    energy = 0.0
    forces = np.zeros((atoms.num_atoms,3))
    stress = np.zeros((3,3))
    try:
     ase_atoms=atoms.ase_converter()
     ase_atoms.calc = calculator #M3GNetCalculator(potential=potential)
     forces = ase_atoms.get_forces()
     energy = ase_atoms.get_potential_energy()+rescale_factor
     stress = ase_atoms.get_stress()
    except:
      print ('Failed for',atoms)
      pass
    return energy,forces,stress



df = pd.DataFrame(data('alignn_ff_db'))
print (df)
for i in glob.glob('../../../dataset/AI/MLFF/alignn_ff_db_energy.json.zip*'):
    fname_e='AI-MLFF-energy-'+i.split('/')[-1].split('_energy.json.zip')[0]+'-test-mae.csv'
    fname_f='AI-MLFF-forces-'+i.split('/')[-1].split('_energy.json.zip')[0]+'-test-multimae.csv'
    #fname_s='AI-MLFF-stresses-'+i.split('/')[-1].split('_energy.json.zip')[0]+'-test-multimae.csv'
    f_e=open(fname_e,'w')
    f_f=open(fname_f,'w')
    #f_s=open(fname_s,'w')

    f_e.write('id,target,prediction\n')
    f_f.write('id,prediction\n')
    #f_s.write('id,prediction\n')


    print (i)
    dat=json.loads(zipfile.ZipFile(i).read(i.split('/')[-1].split('.zip')[0]))
    print(dat['test'])
    count=0
    for key,val in dat['test'].items():
        entry = df[df['jid']==key]
        atoms=Atoms.from_dict(entry.atoms.values[0])
        #print(key,val,df[df['jid']==key],atoms)
        energy,forces,stress=get_alignn_forces(atoms)
        energy=energy/atoms.num_atoms
        count+=1
        print (count,key,val,energy)
        line=key+','+str(val)+','+str(energy)+'\n'
        f_e.write(line)
        line=key+','+str(';'.join(map(str,np.array(forces).flatten())))+'\n'
        f_f.write(line)
        #line=key+','+str(';'.join(map(str,np.array(stress).flatten())))+'\n'
        #f_s.write(line)
    f_e.close()
    f_f.close()
    #f_s.close()
    cmd = 'zip '+fname_e+'.zip '+fname_e
    #os.system(cmd)
    cmd = 'zip '+fname_f+'.zip '+fname_f
    os.system(cmd)
    #cmd = 'zip '+fname_s+'.zip '+fname_s
    #os.system(cmd)
    cmd='rm '+fname_e
    #os.system(cmd)
    cmd='rm '+fname_f
    #os.system(cmd)
    #cmd='rm '+fname_s
    #os.system(cmd)
    #break

