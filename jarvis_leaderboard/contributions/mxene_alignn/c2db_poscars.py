from ase.db import connect
import ase
import pandas as pd


'''
The Computational 2D Materials Database: High-Throughput Modeling and Discovery of Atomically Thin Crystals
Sten Haastrup, Mikkel Strange, Mohnish Pandey, Thorsten Deilmann, Per S. Schmidt, Nicki F. Hinsche, Morten N. Gjerding, Daniele Torelli, Peter M. Larsen, Anders C. Riis-Jensen, Jakob Gath, Karsten W. Jacobsen, Jens Jørgen Mortensen, Thomas Olsen, Kristian S. Thygesen
2D Materials 5, 042002 (2018)

Recent Progress of the Computational 2D Materials Database (C2DB)
M. N. Gjerding, A. Taghizadeh, A. Rasmussen, S. Ali, F. Bertoldo, T. Deilmann, U. P. Holguin, N. R. Knøsgaard, M. Kruse, A. H. Larsen, S. Manti, T. G. Pedersen, T. Skovhus, M. K. Svendsen, J. J. Mortensen, T. Olsen, K. S. Thygesen
2D Materials 8, 044002 (2021)
'''

# Connect to database
db = connect('c2db-xxxx.db') #earlier version was public on https://cmr.fysik.dtu.dk/c2db/c2db.html now it says "provided upon request". So if have access is good to use this code, otherwise poscars are provided as .zip file and formation energy energy in id_prop.csv

rows = db.select('class=MXene', sort='gap')

i=0
data=[]
for row in rows:
   i = i + 1
   formula=row.formula
   fo=row.hform
   gap=row.gap
   ene=row.energy
   label=formula
   name='POSCAR-{}.vasp'.format(label)
   ase.io.write('POSCAR-{}.vasp'.format(label),row.toatoms())
   data.append([name,label,fo,ene,gap])
  
data_df=pd.DataFrame(data,columns=['pos-name','formula','hform','ene','gap'])
data_df.to_excel('data_mxene.xlsx',index=None)
