import os,glob
for i in glob.glob("*.csv"):
  cmd = 'zip '+i+'.zip '+i
  os.system(cmd)
