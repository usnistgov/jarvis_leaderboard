import zipfile
import json
import glob
import pandas as pd
import numpy as np
from jarvis.core.atoms import Atoms
import os
from alignn.ff.ff import AlignnAtomwiseCalculator,default_path,ForceField
model_path = default_path()
calc = AlignnAtomwiseCalculator(path=model_path)
#export CUDA_VISIBLE_DEVICES=""
#wget https://figshare.com/ndownloader/files/40357663 -O mlearn.json.zip
 
def get_alignn_forces(atoms=[]):
    
    ff = ForceField(
        jarvis_atoms=atoms,
        model_path=model_path,
    )
    en, fs = ff.unrelaxed_atoms()
    return en,fs
    #return en*atoms.num_atoms,fs*atoms.num_atoms


df = pd.DataFrame(json.loads(zipfile.ZipFile('mlearn.json.zip').read('mlearn.json')))
print (df)
for i in glob.glob('../../dataset/AI/MLFF/*energy*.zip'):
 if 'mlearn' in i:
    fname_e='AI-MLFF-energy-'+i.split('/')[-1].split('_energy.json.zip')[0]+'-test-mae.csv'
    fname_f='AI-MLFF-forces-'+i.split('/')[-1].split('_energy.json.zip')[0]+'-test-multimae.csv'
    #fname_s='AI-MLFF-stresses-'+i.split('/')[-1].split('_energy.json.zip')[0]+'-test-multimae.csv'
    f_e=open(fname_e,'w')
    f_f=open(fname_f,'w')
    #f_s=open(fname_s,'w')

    f_e.write('id,prediction\n') 
    f_f.write('id,prediction\n') 
    #f_s.write('id,prediction\n') 


    print (i)
    dat=json.loads(zipfile.ZipFile(i).read(i.split('/')[-1].split('.zip')[0]))
    print(dat['test']) 
    for key,val in dat['test'].items():   
        entry = df[df['jid']==key]
        atoms=Atoms.from_dict(entry.atoms.values[0])
        #print(key,val,df[df['jid']==key],atoms)
        energy,forces=get_alignn_forces(atoms)
        #energy,forces,stress=get_alignn_forces(atoms)
        print (key,val,energy)
        line=key+','+str(energy)+'\n'
        f_e.write(line)
        line=key+','+str(';'.join(map(str,np.array(forces).flatten())))+'\n'
        f_f.write(line)
        #line=key+','+str(';'.join(map(str,np.array(stress).flatten())))+'\n'
        #f_s.write(line)
    f_e.close()
    f_f.close()
    #f_s.close()   
    cmd = 'zip '+fname_e+'.zip '+fname_e
    os.system(cmd)
    cmd = 'zip '+fname_f+'.zip '+fname_f
    os.system(cmd)
    #cmd = 'zip '+fname_s+'.zip '+fname_s
    #os.system(cmd)
    cmd='rm '+fname_e
    os.system(cmd)
    cmd='rm '+fname_f
    os.system(cmd)
    #cmd='rm '+fname_s
    #os.system(cmd)
    break


