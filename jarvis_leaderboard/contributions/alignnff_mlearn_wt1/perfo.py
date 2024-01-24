import glob
from jarvis_leaderboard.rebuild import get_metric_value,get_results
for ii in glob.glob('jarvis_leaderboard/jarvis_leaderboard/contributions/alignnff_mlearn_nisab_1/*.csv.zip'):
  #if 'multimae' in ii:
     bname=ii.split('/')[-1]
     print(bname)
     names,vals=get_results(bench_name=bname)
     for i,j in zip(names,vals):
         print(i,j)
     print()
     print()
#names,vals=get_results(bench_name='AI-MLFF-energy-mlearn_Mo-test-mae.csv.zip')
##for i,j in zip(names,vals):
# print(i,j)
#
