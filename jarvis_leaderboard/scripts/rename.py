import pandas as pd
import glob,os
x=[]
cwd=os.getcwd()
cmds=[]
for i in glob.glob('*/*csv.zip'):
    #print (i)
    dir=i.split('/')[0]
    old_name=i.split('/')[1]
    os.chdir(dir)
    if os.path.exists(old_name):
      tmp=old_name.split('-')
      new_name=tmp[4]+'-'+tmp[0]+'-'+tmp[2]+'-'+tmp[3]+'-'+tmp[1]+'-'+tmp[5]+'.csv'
      print (old_name,new_name)
      df=pd.read_csv(old_name)
      df.to_csv(new_name,index=False)
      cmd='zip '+new_name+'.zip '+new_name
      os.system(cmd)
      cmd='rm '+old_name
      os.system(cmd)       
      print (cmd)
      cmds.append(cmd)
    os.chdir(cwd)

print (cmds)
    
