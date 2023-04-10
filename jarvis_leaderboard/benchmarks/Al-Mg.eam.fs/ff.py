from jarvis.db.figshare import data
import pandas as pd
import os

d=data('jff')
df=pd.DataFrame(d)
dff = df[df['ref']=='mp-134']
#len(dff)


count=0
for i,ii in dff[['func','elastic_tensor_data','jid']].iterrows():
    try:
        func=ii['func']
        et=ii['elastic_tensor_data']
        print (func,et['voigt_bulk_modulus'],ii['jid'])
        fname='SinglePropertyPrediction-test-bulk_modulus_JVASP_816_Al-dft_3d-FF-mae.csv'
        f=open(fname,'w')
        f.write('id,prediction\n')
        line='JVASP-816,'+str(et['voigt_bulk_modulus'])
        f.write(line)
        f.close()
        cmd='zip '+fname+'.zip '+fname
        os.system(cmd)
        cmd='rm '+fname
        os.system(cmd)
        if not os.path.exists(func):
           os.makedirs(func)
        cmd='mv '+fname+'.zip '+func
        os.system(cmd)
        cmd='cp ff.py '+func
        os.system(cmd)
        cmd='cp metadata.json '+func
        os.system(cmd)
              
        count+=1
    except:
        pass
print(count)



