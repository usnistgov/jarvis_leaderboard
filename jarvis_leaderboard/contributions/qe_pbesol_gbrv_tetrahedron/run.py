

import numpy as np
import sys 
import copy
import os


dict_amu = {\
'H':   1.00794,\
'He':  4.002602,\
'Li':  6.941,\
'Be':  9.012182,\
'B':   10.811,\
'C':   12.0107,\
'N':   14.0067,\
'O':   15.9994,\
'F':   18.9984032,\
'Ne':  20.1797,\
'Na':  22.98976928,\
'Mg':  24.3050,\
'Al':  26.9815386,\
'Si':  28.0855,\
'P':   30.973762,\
'S':   32.065,\
'Cl':  35.453,\
'Ar':  39.948,\
'K':   39.0983,\
'Ca':  40.078,\
'Sc':  44.955912,\
'Ti':  47.867,\
'V':   50.9415,\
'Cr':  51.9961,\
'Mn':  54.938045,\
'Fe':  55.845,\
'Co':  58.933195,\
'Ni':  58.6934,\
'Cu':  63.546,\
'Zn':  65.409,\
'Ga':  69.723,\
'Ge':  72.64,\
'As':  74.92160,\
'Se':  78.96,\
'Br':  79.904,\
'Kr':  83.798,\
'Rb':  85.4678,\
'Sr':  87.62,\
'Y':   88.90585,\
'Zr':  91.224,\
'Nb':  92.906,\
'Mo':  95.94,\
'Tc':  98,\
'Ru':  101.07,\
'Rh':  102.905,\
'Pd':  106.42,\
'Ag':  107.8682,\
'Cd':  112.411,\
'In':  114.818,\
'Sn':  118.710,\
'Sb':  121.760,\
'Te':  127.60,\
'I':   126.904,\
'Xe':  131.293,\
'Cs':  132.9054519,\
'Ba':  137.327,\
'La':  138.90547,\
'Ce':  140.116,\
'Pr':  140.90765,\
'Nd':  144.242,\
'Pm':  145,\
'Sm':  150.36,\
'Eu':  151.964,\
'Gd':  157.25,\
'Tb':  158.92535,\
'Dy':  162.500,\
'Ho':  164.930,\
'Er':  167.259,\
'Tm':  168.93421,\
'Yb':  173.04,\
'Lu':  174.967,\
'Hf':  178.49,\
'Ta':  180.94788,\
'W':   183.84,\
'Re':  186.207,\
'Os':  190.23,\
'Ir':  192.217,\
'Pt':  195.084,\
'Au':  196.966569,\
'Hg':  200.59,\
'Tl':  204.3833,\
'Pb':  207.2,\
'Bi':  208.98040}


def check_frac(n):
    for f in [0.0, 0.3333333333333333, 0.25 ,0.5 ,0.75 ,0.6666666666666667 ,1.0, 1.5, 2.0, -0.5, -2.0,-1.5,-1.0,  1.0/2.0**0.5, -1.0/2.0**0.5, 3.0**0.5/2.0, -3.0**0.5/2.0, 1.0/3.0**0.5, -1.0/3.0**0.5, 1.0/2.0/3**0.5, -1.0/2.0/3**0.5]:
        if abs(f-n) < 2e-4:
            return f
        if abs(-f-n) < 2e-4:
            return -f
        if abs(f) > 1e-5:
            if abs(1.0/f-n) < 2e-4:
                return 1.0/f
            if abs(-1.0/f-n) < 2e-4:
                return -1.0/f

    return n

def go(directory, poscar, nk, nq, nk2):

    print(directory)
    print(poscar)
    

    file_CONTCAR = open(directory+"/"+poscar,'r')

    file_CONTCAR.readline()
    file_CONTCAR.readline()

    A = np.zeros((3,3))

    A[0,:] = [float(i) for i in file_CONTCAR.readline().split()]
    A[1,:] = [float(i) for i in file_CONTCAR.readline().split()]
    A[2,:] = [float(i) for i in file_CONTCAR.readline().split()]

    for i in range(3):
        for j in range(3):
            A[i,j] = check_frac(A[i,j])
    elements = file_CONTCAR.readline().split()
    numel =  [int(i) for i in file_CONTCAR.readline().split()]

    ntyp = str(len(elements))

    el_list = []
    for e,n in zip(elements, numel):
        for m in range(n):
            el_list.append(e)


    file_CONTCAR.readline()


    ntot = np.sum(numel)
    pos = np.zeros((ntot,3),dtype=float)
    for i in range(ntot):
        pos[i,:] = [float(i) for i in file_CONTCAR.readline().split()]
        for j in range(3): #neatin                                                                                                                                                                                                                                                                                           
            pos[i,j] = check_frac(pos[i,j])



    print(A)
    
    print(pos)
    
    psp ="ATOMIC_SPECIES\n"
    for el in elements:
        psp += el + "   " + str( dict_amu[el] ) + "   " + el.lower()+".pbesol.UPF\n"

    cell ="CELL_PARAMETERS angstrom\n"
    for i in range(3):
        cell += str(A[i,0])+"  " +str(A[i,1]) + "  "+str(A[i,2])+"\n"

    atoms ="ATOMIC_POSITIONS crystal\n"
    for e,i in zip(el_list, range(ntot)):
        atoms += e + "   "+str(pos[i,0])+"  " +str(pos[i,1]) + "  "+str(pos[i,2])+"\n"

    temp_scf = open("templates/al.scf.in", "r")
    scf=""

    temp_ph = open("templates/al.ph.in", "r")
    ph=""

    temp_elph = open("templates/al.elph.in", "r")
    elph=""

    nks = [str(i) for i in nk]
    nqs = [str(i) for i in nq]
    nk2s = [str(i) for i in nk2]

    prefix = "qe_"+nks[0]+"_"+nks[1]+"_"+nks[2]+"__"+nqs[0]+"_"+nqs[1]+"_"+nqs[2]+"__"+nk2s[0]+"_"+nk2s[1]+"_"+nk2s[2]

    for line in temp_scf:
        t = copy.copy(line)

        t = t.replace("NAT", str(ntot))
        t = t.replace("NTYP", str(ntyp))
        t = t.replace("PREFIX", prefix)

        t = t.replace("NK1", str(nk[0]))
        t = t.replace("NK2", str(nk[1]))
        t = t.replace("NK3", str(nk[2]))

        scf += t

    scf += psp
    scf += cell
    scf += atoms


    for line in temp_ph:
        t = copy.copy(line)

#        t = t.replace("NAT", str(ntot))
        t = t.replace("PREFIX", prefix)

        t = t.replace("NQ1", str(nq[0]))
        t = t.replace("NQ2", str(nq[1]))
        t = t.replace("NQ3", str(nq[2]))

        ph += t


    for line in temp_elph:
        t = copy.copy(line)

#        t = t.replace("NAT", str(ntot))
        t = t.replace("PREFIX", prefix)

        t = t.replace("NQ1", str(nq[0]))
        t = t.replace("NQ2", str(nq[1]))
        t = t.replace("NQ3", str(nq[2]))

        t = t.replace("NEP1", str(nk2[0]))
        t = t.replace("NEP2", str(nk2[1]))
        t = t.replace("NEP3", str(nk2[2]))

        elph += t

        
    mypath=directory+"/"+prefix
    if not os.path.exists(mypath):
         os.makedirs(mypath)

    temp_ph.close()
    temp_elph.close()
    temp_scf.close()

    of_scf = open(mypath + "/qe.scf.in", "w")
    of_ph = open(mypath + "/qe.ph.in", "w")
    of_elph = open(mypath + "/qe.elph.in", "w")

    of_scf.write(scf)
    of_ph.write(ph)
    of_elph.write(elph)

    of_scf.close()
    of_ph.close()
    of_elph.close()

    os.system("cp templates/job.temp " + mypath)

    print("cd "+mypath+"; sbatch job.temp; cd /home/kfg/output/supercond")
    os.system("cd "+mypath+"; sbatch job.temp; cd /home/kfg/output/supercond")

    

#    print(psp)
#   print(cell)
#   print(atoms)
    
    
for x in ["JVASP-14615","JVASP-14837","JVASP-19684","JVASP-19821","JVASP-934","JVASP-961", "JVASP-14590"]:
    
    go(x, "POSCAR_cif_relaxed", [18,18,18],[2,2,2],[12,12,12])
    go(x, "POSCAR_cif_relaxed", [8,8,8],[2,2,2],[12,12,12])

#    go(x, "POSCAR_cif_relaxed", [8,8,8],[2,2,2],[12,12,12])

    go(x, "POSCAR_cif_relaxed", [8,8,8],[3,3,3],[12,12,12])
    go(x, "POSCAR_cif_relaxed", [8,8,8],[2,2,2],[18,18,18])

#    go(x, "POSCAR_cif_relaxed", [18,18,18],[2,2,2],[18,18,18])
#
    go(x, "POSCAR_cif_relaxed", [12,12,12],[2,2,2],[12,12,12])
    go(x, "POSCAR_cif_relaxed", [12,12,12],[2,2,2],[18,18,18])
    go(x, "POSCAR_cif_relaxed", [12,12,12],[2,2,2],[24,24,24])
    go(x, "POSCAR_cif_relaxed", [12,12,12],[2,2,2],[36,36,36])

    go(x, "POSCAR_cif_relaxed", [12,12,12],[3,3,3],[12,12,12])
    go(x, "POSCAR_cif_relaxed", [12,12,12],[3,3,3],[18,18,18])
    go(x, "POSCAR_cif_relaxed", [12,12,12],[3,3,3],[24,24,24])
    go(x, "POSCAR_cif_relaxed", [12,12,12],[3,3,3],[36,36,36])

    go(x, "POSCAR_cif_relaxed", [12,12,12],[4,4,4],[12,12,12])
    go(x, "POSCAR_cif_relaxed", [12,12,12],[4,4,4],[18,18,18])
    go(x, "POSCAR_cif_relaxed", [12,12,12],[4,4,4],[24,24,24])
    go(x, "POSCAR_cif_relaxed", [12,12,12],[4,4,4],[36,36,36])

    go(x, "POSCAR_cif_relaxed", [12,12,12],[6,6,6],[12,12,12])
    go(x, "POSCAR_cif_relaxed", [12,12,12],[6,6,6],[18,18,18])
    go(x, "POSCAR_cif_relaxed", [12,12,12],[6,6,6],[24,24,24])
    go(x, "POSCAR_cif_relaxed", [12,12,12],[6,6,6],[36,36,36])

