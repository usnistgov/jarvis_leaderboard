import os
import pandas as pd
from jarvis.db.figshare import data,get_jid_data
from jarvis.db.jsonutils import dumpjson

fname='SinglePropertyPrediction-test-bandgap-dft_3d-ES-mae.csv'
fname="SinglePropertyPrediction-test-Tc_supercon-dft_3d-ES-mae.csv"
fname='SinglePropertyPrediction-test-bulk_modulus-dft_3d-ES-mae.csv'
fname='SinglePropertyPrediction-test-bandgap-dft_3d-ES-mae.csv'
fname='SinglePropertyPrediction-test-slme-dft_3d-ES-mae.csv'
fname='SinglePropertyPrediction-test-epsx-dft_3d-ES-mae.csv'
fname='SinglePropertyPrediction-test-epsx-dft_3d-ES-mae.csv'
fname='Spectra-test-dielectric_function-dft_3d-ES-multimae.csv'
#pp='bandgap'
#pp='Tc_supercon'
#pp='bulk_modulus'
#pp='bandgap'
pp=fname.split('-')[2]
json_name=''
df=pd.read_csv(fname)
for i,ii in df.iterrows():
    print (ii)
    formula=get_jid_data(dataset='dft_3d',jid=ii['id'])['formula']
    name=pp+'_'+str(ii['id']).replace('-','_')+'_'+formula
    new_csv=fname.replace(pp,name)
    #new_csv=fname.replace('-dft_3d','-'+name)
    f=open(new_csv,'w')
    line='id,prediction\n'
    f.write(line)
    line=str(ii['id'])+','+str(ii['prediction'])+'\n'
    f.write(line)
    f.close()

    tmp=new_csv.split('-')
    print ('tmp',tmp)
    #json_name='dft_3d_'+tmp[2]+'_'+tmp[3].split('dft_3d_')[1]+'.json'
    #json_name=tmp[2]+'.json'
    json_name=tmp[3]+'_'+tmp[2]+'.json'
    print ('json_name',json_name)
    #md_name=tmp[2]+'_'+tmp[3].split('dft_3d_')[1]+'.md'
    md_name=tmp[2]+'.md'
    #md_name=tmp[2].replace('dft_3d_','')+'.md'
    print ('md_name',md_name)
    content='# Model for XYZ\n\n<h2>Model benchmarks</h2>\n<table style="width:100%" id="j_table">\n <thead>\n  <tr>\n<th>Model name</th>\n    <th>Dataset</th>\n   <!-- <th>Method</th>-->\n    <th>MAE</th>\n    <th>Team name</th>\n    <th>Dataset size</th>\n    <th>Date submitted</th>\n    <th>Notes</th>\n  </tr>\n </thead>\n<!--table_content-->\n</table>\n'
    #content=content.replace('XYZ',tmp[2]+'_'+tmp[3].split('dft_3d_')[1])
    content=content.replace('XYZ',tmp[2].replace('dft_3d_',''))
    f=open(md_name,'w')
    f.write(content)
    f.close()

    info={}
    info['train']={}
    if 'target' in ii:
        info['test']={str(ii['id']):ii['target']}
    dumpjson(data=info,filename=json_name)
    cmd='zip '+json_name+'.zip '+json_name
    os.system(cmd)
    cmd='rm '+json_name
    os.system(cmd)
  
    cmd='zip '+new_csv+'.zip '+new_csv
    os.system(cmd)
    cmd='rm '+new_csv
    os.system(cmd)
  
    #print (new_csv)
    #print()
    #break
#print (df)

