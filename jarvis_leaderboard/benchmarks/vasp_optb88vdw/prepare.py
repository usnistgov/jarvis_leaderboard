from jarvis.db.jsonutils import loadjson
from jarvis.db.figshare import data
p=['dft_3d_bulk_modulus.json','bulk_modulus_kv']
keys=['dft_3d_bandgap.json','optb88vdw_bandgap']
d=loadjson(keys[0])
dft=data('dft_3d')
#>>> d['test']
#{'JVASP-867': 142, 'JVASP-91': 443, 'JVASP-1002': 99.2, 'JVASP-890': 75.8, 'JVASP-14606': 109, 'JVASP-963': 195, 'JVASP-984': 269, 'JVASP-25065': 13.3, 'JVASP-25114': 3.7, 'JVASP-978': 2.9, 'JVASP-25180': 18.4, 'JVASP-1011': 12.4, 'JVASP-14604': 9.3, 'JVASP-816': 79.4, 'JVASP-1130': 69.8, 'JVASP-23864': 35.4, 'JVASP-23862': 26.6, 'JVASP-20326': 51.4, 'JVASP-116': 165, 'JVASP-182': 225, 'JVASP-1174': 75.6}
p=[]
f=open('dat.csv','w')
line='id,target,prediction\n'
f.write(line)

for i,j in d['test'].items():
    jid=i
    expv=j
    for k in dft:
       if k['jid']==jid:
          q=keys[1]
          line=jid+','+str(expv)+','+str(k[q])+'\n'
          f.write(line)
f.close() 
print (len(d['test']))
#for i in dft:
# if i['jid'] in list(d['test'].keys()):
#     print (i['jid'],i['bulk_modulus_kv'])

