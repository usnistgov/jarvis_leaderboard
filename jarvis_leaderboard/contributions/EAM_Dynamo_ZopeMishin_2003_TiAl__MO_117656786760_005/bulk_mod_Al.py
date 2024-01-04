from pathlib import Path
from zipfile import ZipFile

import iprPy

import atomman as am
import atomman.unitconvert as uc

###############################################################################

# Specify the material as known in JARVIS and IPR
family = 'A1--Cu--fcc'
element = 'Al'
jarvis_number = 816

# Specify the LAMMPS command to use
lammps_command = '/home/lmh1/LAMMPS/2022-06-23/src/lmp_serial'

# Other calculation settings
strainrange = 1e-8

###############################################################################

# Auto-select potential ID based on folder name
potential_LAMMPS_id = Path.cwd().name

# Load calculation
calc = iprPy.load_calculation('elastic_constants_static')

# Fetch potential and crystal from database
potential = am.load_lammps_potential(id=potential_LAMMPS_id, getfiles=True)
ucell = am.load('crystal', potential=potential, family=family,
                symbols=element, standing='good', method='dynamic')

# Run the calculation and get bulk modulus
results = calc.calc(lammps_command, ucell, potential, strainrange=strainrange, ftol=1e-6)
bulk_modulus = uc.get_in_units(results['C'].bulk('Voigt'), 'GPa')

# Save in the JARVIS leaderboard format
csvname = f'FF-SinglePropertyPrediction-bulk_modulus_JVASP_{jarvis_number}_{element}-dft_3d-test-mae.csv'
zipname = csvname + '.zip'

contents = f'id,prediction\nJVASP-{jarvis_number},{bulk_modulus}\n'
with ZipFile(zipname, 'w') as myzip:
    myzip.writestr(csvname, contents)
