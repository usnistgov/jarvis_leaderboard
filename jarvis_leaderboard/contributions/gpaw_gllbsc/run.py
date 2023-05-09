#!/usr/bin/env python
import pandas as pd
from gpaw import GPAW, PW, FermiDirac
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
from jarvis.core.kpoints import Kpoints3D
import time
from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac
#source: https://wiki.fysik.dtu.dk/gpaw/tutorialsexercises/electronic/band_gap/band_gap.html


def get_band_gap(atoms=None, cutoff=500,kpts=[7,7,7]):
    calc = GPAW(mode=PW(cutoff),
                xc='GLLBSC',
                kpts=kpts,  # Choose and converge carefully!
                occupations=FermiDirac(0.01),
                txt='gs.out')
    atoms.calc = calc
    atoms.get_potential_energy()
    # Calculate the discontinuity potential and the discontinuity
    homo, lumo = calc.get_homo_lumo()
    response = calc.hamiltonian.xc.response
    dxc_pot = response.calculate_discontinuity_potential(homo, lumo)
    KS_gap, dxc = response.calculate_discontinuity(dxc_pot)

    # Fundamental band gap = Kohn-Sham band gap + derivative discontinuity
    QP_gap = KS_gap + dxc

    print(f'Kohn-Sham band gap:         {KS_gap:.2f} eV')
    print(f'Discontinuity from GLLB-sc: {dxc:.2f} eV')
    print(f'Fundamental band gap:       {QP_gap:.2f} eV')
    return QP_gap,KS_gap,dxc



# atoms = bulk('Si', 'diamond', 5.431)
# QP_gap,KS_gap,dxc = get_band_gap(atoms=atoms)


df=pd.read_csv('https://github.com/usnistgov/jarvis_leaderboard/raw/main/jarvis_leaderboard/contributions/vasp_tbmbj/ES-SinglePropertyPrediction-bandgap-dft_3d-test-mae.csv.zip')


for i,ii in df.iterrows():
    try:
        jid = ii['id']
        print('jid',jid)
        dat=get_jid_data(dataset='dft_3d',jid=jid)
        atoms=Atoms.from_dict(dat['atoms'])
        ase_atoms=atoms.ase_converter()
        kp = Kpoints3D().automatic_length_mesh(

            lattice_mat=dat['atoms']['lattice_mat'],
            length=dat["kpoint_length_unit"],
        )
        kpts = kp._kpoints[0]
        t1=time.time()
        QP_gap,KS_gap,dxc = get_band_gap(atoms=ase_atoms,kpts=kpts)
        t2=time.time()
        name=jid.replace('-','_')+'_'+atoms.composition.reduced_formula
        fname='ES-SinglePropertyPrediction-bandgap_'+name+'-dft_3d-test-mae.csv'
        f=open(fname,'w')
        line='id,prediction\n'
        f.write(line)
        print('jid,QP_gap,KS_gap,dxc',jid,QP_gap,KS_gap,dxc,t2-t1)
        line=jid+','+str(QP_gap)+'\n'
        f.write(line)
        f.close()
    except:
       print('Failed for jid', jid)
       pass

       


# In[ ]:




