import glob
import os
import pandas as pd
cwd=os.getcwd()
for i in glob.glob('*/TextClass-test-categories-arXiv-AI-acc.csv.zip'):
    dr=i.split('/')[0]
    os.chdir(dr)
    df=pd.read_csv(i.split('/')[1])
    df_new = df.drop_duplicates(subset='id')
    df_new.to_csv(i.split('/')[1].split('.zip')[0],index=False)
    cmd = 'zip '+i.split('/')[1]+' '+i.split('/')[1].split('.zip')[0]
    print (cmd)
   
    os.system(cmd)
    os.chdir(cwd)
    print (i)
