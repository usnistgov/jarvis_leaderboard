#! /usr/bin/env python
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
from jarvis.core.kpoints import Kpoints3D
from nexus import settings, job, run_project, obj
from nexus import generate_physical_system
from nexus import generate_pwscf, generate_pw2qmcpack
from nexus import generate_qmcpack, vmc, loop, linear, dmc
from nexus import read_structure
from numpy import mod, sqrt, array
from qmcpack_input import spindensity
import numpy as np
from structure import optimal_tilematrix
from numpy.linalg import det
import os


jid='JVASP-1002'
pseudo_path = "/wrk/dtw2/nn-jastrow/pseudopotentials"
settings(
    results="./results-"+jid,
    pseudo_dir=pseudo_path,  # location of pseudopotential directory
    sleep=1,
    runs="./runs-"+jid,
    machine="raritan",  # machine
)

pseudo_Zeff = dict(
    Ag = 19,
    Al = 3,
    Ar = 8,
    As = 5,
    Au = 19,
    B = 3,
    Be = 2,
    Bi = 5,
    Br = 7,
    C = 4,
    Ca = 10,
    Cl = 7,
    Co = 17,
    Cr = 14,
    Cu = 19,
    F = 7,
    Fe = 16,
    Ga = 3,
    Ge = 4,
    H = 1,
    He = 2,
    I = 7,
    Ir = 17,
    K = 9,
    Kr = 8,
    Li = 1,
    Mg = 2,
    Mn = 15,
    Mo = 14,
    N = 5,
    Na = 1,
    Ne = 8,
    Ni = 18,
    O = 6,
    P = 5,
    Pd = 18,
    S = 6,
    Sc = 11,
    Se = 6,
    Si = 4,
    Tb = 19,
    Te = 6,
    Ti = 12,
    V = 13,
    W = 14,
    Zn = 20,
    )
    
pseudo_dict = {
    "Ag": "Ag.ccECP.AREP.upf", #Ag.ccECP.SOREP.upf
    "Al": "Al.ccECP.upf", 
    "Ar": "Ar.ccECP.upf", 
    "As": "As.ccECP.upf",
    "Au": "Au.ccECP.AREP.upf", #Au.ccECP.SOREP.upf
    "B": "B.ccECP.upf",
    "Be": "Be.ccECP.upf",
    "Bi": "Bi.ccECP.AREP.upf", #Bi.ccECP.SOREP.upf
    "Br": "Br.ccECP.upf",
    "C": "C.ccECP.upf",
    "Ca": "Ca.ccECP.upf",
    "Cl": "Cl.ccECP.upf",
    "Co": "Co.ccECP-soft.upf", #Co.opt.upf
    "Cr": "Cr.ccECP.upf", #Cr.opt.upf
    "Cu": "Cu.ccECP-soft.upf", #Cu.opt.upf
    "F": "F.ccECP.upf",
    "Fe": "Fe.ccECP-soft.upf", #Fe.opt.upf
    "Ga": "Ga.ccECP.upf",
    "Ge": "Ge.ccECP.upf",
    "H": "H.ccECP.upf",
    "He": "He.ccECP.upf",
    "I": "I.ccECP.AREP.upf", #I.ccECP.SOREP.upf
    "Ir": "Ir.ccECP.AREP.upf", #Ir.ccECP.SOREP.upf
    "K": "K.ccECP.upf",
    "Kr": "Kr.ccECP.upf",
    "Li": "Li.ccECP.upf",
    "Mg": "Mg.ccECP.upf",
    "Mn": "Mn.ccECP.upf", #Mn.opt.upf
    "Mo": "Mo.ccECP.AREP.upf", #Mo.ccECP.SOREP.upf
    "N": "N.ccECP.upf",
    "Na": "Na.ccECP.upf",
    "Ne": "Ne.ccECP.upf",
    "Ni": "Ni.ccECP-soft.upf", #Ni.opt.upf
    "O": "O.ccECP.upf",
    "P": "P.ccECP.upf",
    "Pd": "Pd.ccECP.AREP.upf", #Pd.ccECP.SOREP.upf
    "S": "S.ccECP.upf",
    "Sc": "Sc.ccECP-soft.upf", #Sc.opt.upf
    "Se": "Se.ccECP.upf",
    "Si": "Si.ccECP.upf",
    "Tb": "Tb.ccECP.AREP.upf", #Tb.ccECP.SOREP.upf
    "Te": "Te.ccECP.AREP.upf", #Te.ccECP.SOREP.upf
    "Ti": "Ti.ccECP-soft.upf", #Ti.opt.upf
    "V": "V.ccECP-soft.upf", #V.opt.upf
    "W": "W.ccECP.AREP.upf", #W.ccECP.SOREP.upf
    "Zn": "Zn.ccECP-soft.upf" #Zn.opt.upf
    }

pseudo_dict_qmc = {
    "Ag": "Ag.ccECP.AREP.xml", #Ag.ccECP.SOREP.xml
    "Al": "Al.ccECP.xml", 
    "Ar": "Ar.ccECP.xml", 
    "As": "As.ccECP.upf",
    "Au": "Au.ccECP.AREP.xml", #Au.ccECP.SOREP.xml
    "B": "B.ccECP.xml",
    "Be": "Be.ccECP.xml",
    "Bi": "Bi.ccECP.AREP.xml", #Bi.ccECP.SOREP.xml
    "Br": "Br.ccECP.xml",
    "C": "C.ccECP.xml",
    "Ca": "Ca.ccECP.xml",
    "Cl": "Cl.ccECP.xml",
    "Co": "Co.ccECP-soft.xml", #Co.opt.xml
    "Cr": "Cr.ccECP.xml", #Cr.opt.xml
    "Cu": "Cu.ccECP-soft.xml", #Cu.opt.xml
    "F": "F.ccECP.xml",
    "Fe": "Fe.ccECP-soft.xml", #Fe.opt.xml
    "Ga": "Ga.ccECP.xml",
    "Ge": "Ge.ccECP.xml",
    "H": "H.ccECP.xml",
    "He": "He.ccECP.xml",
    "I": "I.ccECP.AREP.xml", #I.ccECP.SOREP.xml
    "Ir": "Ir.ccECP.AREP.xml", #Ir.ccECP.SOREP.xml
    "K": "K.ccECP.xml",
    "Kr": "Kr.ccECP.xml",
    "Li": "Li.ccECP.xml",
    "Mg": "Mg.ccECP.xml",
    "Mn": "Mn.ccECP.xml", #Mn.opt.xml
    "Mo": "Mo.ccECP.AREP.xml", #Mo.ccECP.SOREP.xml
    "N": "N.ccECP.xml",
    "Na": "Na.ccECP.xml",
    "Ne": "Ne.ccECP.xml",
    "Ni": "Ni.ccECP-soft.xml", #Ni.opt.xml
    "O": "O.ccECP.xml",
    "P": "P.ccECP.upf",
    "Pd": "Pd.ccECP.AREP.xml", #Pd.ccECP.SOREP.xml
    "S": "S.ccECP.xml",
    "Sc": "Sc.ccECP-soft.xml", #Sc.opt.xml
    "Se": "Se.ccECP.xml",
    "Si": "Si.ccECP.xml",
    "Tb": "Tb.ccECP.AREP.xml", #Tb.ccECP.SOREP.xml
    "Te": "Te.ccECP.AREP.xml", #Te.ccECP.SOREP.xml
    "Ti": "Ti.ccECP-soft.xml", #Ti.opt.xml
    "V": "V.ccECP-soft.xml", #V.opt.xml
    "W": "W.ccECP.AREP.xml", #W.ccECP.SOREP.xml
    "Zn": "Zn.ccECP-soft.xml" #Zn.opt.xml
    }

info = get_jid_data(jid=jid, dataset="dft_3d")
atoms = Atoms.from_dict(info["atoms"])
filename = "POSCAR-" + jid + ".vasp"
atoms.write_poscar(filename)
kplength = info["kpoint_length_unit"]
kp = (
    Kpoints3D()
    .automatic_length_mesh(lattice_mat=atoms.lattice_mat, length=kplength)
    ._kpoints
)
kp = np.array(kp).flatten().tolist()
print("kp", kp)
structure = read_structure(filename, format="poscar")
print(structure)
boundaries = "ppp"
prim=structure
r_ws=prim.rwigner()
r_ws_min=4

volfac=(r_ws_min/prim.rwigner())**3

if volfac < 0.5:
    volfac=1

r_ws_super=prim.rwigner()*np.cbrt(volfac)





print('volfac',volfac)
opt_matrix, opt_ws = optimal_tilematrix(axes = prim.axes, volfac=volfac)
print('opt_ws',opt_ws)
print('opt_matrix',opt_matrix)
print('r_ws',r_ws)
print('r_ws_min',r_ws_min)
print('r_ws_super',r_ws_super)

#supercells=[opt_matrix.tolist()]
supercells = [[[2, 0, 0], [0, 2, 0], [0, 0, 2]]]
print(supercells)
linopt1 = linear(
    energy=0.0,
    unreweightedvariance=1.0,
    reweightedvariance=0.0,
    timestep=0.3,
    samples=20000,
    blocks=100,
    warmupsteps=200,
    #steps=1,
    #stepsbetweensamples=1,
    substeps=5,
    nonlocalpp=False,
    #usebuffer=True,
    minwalkers=0.5,
    #usedrift=True,
    minmethod="OneShiftOnly",
    #shift_i=0.001,
    #shift_s=0.1,
)

j3 = False
# if you want to run VMC (if false you run DMC)
run_vmc = False


# DMC options
dmc_eqblocks = 200
dmcblocks = 300
dt_dmc = 0.01 #DMC Timestep
tmoves = False
##


shared_qe = obj(
    occupations="smearing",
    smearing="gaussian",
    degauss=0.005,
    input_dft="PBE",
    ecut=120,
    #ecut=converg_cutoff(atoms=atoms), #Plane Wave cutoff energy
    conv_thr=1.0e-7,
    mixing_beta=0.2,
    nosym=True,
    use_folded=True,
    spin_polarized=True,
)

qe_presub = 'module load intel/oneapi/base/2021.1.0 intel/oneapi/hpc/2021.1.0 hdf5/1.12.2/oneapi-2021.1.0' #loaded modules
qmcpack_presub = 'module load intel/oneapi/base/2021.1.0 intel/oneapi/hpc/2021.1.0 hdf5/1.12.2/oneapi-2021.1.0'

qe_job = job(hours=168,nodes=4, processes_per_node=1,  threads=16,app='-n 4 --mpi=pmi2 /home/dtw2/qe-6.4-patched-onepai-2021.1.0/bin/pw.x', presub=qe_presub) #submission details
qmc_job = job(hours=168,nodes=8,processes_per_node=1,threads=16,app='-n 8 --mpi=pmi2 /home/dtw2/qmcpack-3.14.0-onepai-2021.1.0/bin/qmcpack_complex', presub=qmcpack_presub)
vmc_dmc_dep = None

scales = [0.95,0.96,0.97,0.98,0.99,1.00,1.01,1.02,1.03,1.04,1.05]  # can modify to have stress/strain

for scale in scales:
    temp = structure.copy()
    temp.stretch(scale, scale, scale)
    system = generate_physical_system(
        structure=temp,
        kshift=(0.5, 0.5, 0.5),
        net_spin=0, #Specify Up - Down Electrons
        **{e:pseudo_Zeff[e] for e in structure.elem}
    ) 
    print('kp',kp[0])
    arr = []
    for sp in atoms.elements:
        arr.append(pseudo_dict[sp])
    scf = generate_pwscf(
        identifier="scf",  # log output goes to scf.out
        path="scf-{}".format(scale),  # directory to run in
        job=qe_job,  # pyscf must run w/o mpi
        system=system,
        input_type="scf",
        pseudos    = arr,
        kgrid=kp,
        wf_collect=False,
        **shared_qe
    )

    for supercell in supercells:
        scell_vol = det(supercell)
        nscf_kgrid_k = int(
            np.ceil((kp[0]/2)/sqrt(scell_vol)) #can be modified in the nscf and DMC stages depending on the number of kpoints needed for nscf convergence and the amount of twists needed for dmc convergence
        )  
        nscf_grid = (nscf_kgrid_k, nscf_kgrid_k, nscf_kgrid_k)
        nscf_system = generate_physical_system(
            structure=temp,
            tiling=supercell,
            kgrid=nscf_grid,
            kshift=(0.5, 0.5, 0.5),
            net_spin=0, #Specify Up - Down Electrons
            **{e:pseudo_Zeff[e] for e in structure.elem}
        )

        nscf = generate_pwscf(
            identifier="nscf",
            path="nscf-{}-{}".format(scale, scell_vol),
            job=qe_job,
            system=nscf_system,
            input_type="nscf",
            pseudos     = arr,
            wf_collect=True,
            dependencies=(scf, "charge_density"),
            **shared_qe
        )

        p2q = generate_pw2qmcpack(
            identifier="p2q",
            path="nscf-{}-{}".format(scale, scell_vol),
            job=job(hours=1,nodes=2, processes_per_node=1,  threads=1,app='-n 1 --mpi=pmi2 /home/dtw2/qe-6.4-patched-onepai-2021.1.0/bin/pw2qmcpack.x',presub=qe_presub),
            write_psir=False,
            dependencies=(nscf, "orbitals"),
        )

        if scale == min(scales):
            super = temp.tile(supercell)
            rcut = super.rwigner() - 0.001

            opt_system = nscf_system
            opt_orbitals = p2q
            opt_name = "bulk"

            linopt1t = linopt1.copy()
            linopt1t.samples *= scell_vol

            linopt2 = linopt1t.copy()
            linopt2.energy = 0.95
            linopt2.unreweightedvariance = 0.05
            linopt2.samples *= 3
            linopt2.minwalkers = 0.5

            linopt3 = linopt2.copy()
            linopt3.samples *= 2
            
            arrqmc = []
            for sp in atoms.elements:
                arrqmc.append(pseudo_dict_qmc[sp])
                
            opt_J2 = generate_qmcpack(
                # block           = True, #Comment if you want to block rest of the calculations
                identifier="opt",
                path="optJ2-{}-{}-{}".format(opt_name, scale, scell_vol),
                job=qmc_job,
                input_type="basic",
                bconds=boundaries,
                system=opt_system,
                pseudos         = arrqmc,
                twistnum=0,
                jastrows=[
                    ("J1", "bspline", 8, rcut),
                    ("J2", "bspline", 8, rcut),
                ],
                spin_polarized=True,
                calculations=[
                    loop(max=6, qmc=linopt1),
                    loop(max=6, qmc=linopt2),
                ],
                dependencies=(opt_orbitals, "orbitals"),
            )

            vmc_dmc_dep = opt_J2
            if j3:
                j3_rcut = min(opt_system["structure"].rwigner(), 4.0)
                opt_J3 = generate_qmcpack(
                    identifier="opt",
                    path="optJ3-{}-{}-{}".format(opt_name, scale, scell_vol),
                    job=qmc_job,
                    input_type="basic",
                    J3=True,
                    spin_polarized=True,
                    twistnum=0,
                    J3_isize=3,
                    J3_esize=3,
                    J3_rcut=j3_rcut,
                    bconds=boundaries,
                    system=opt_system,
                    pseudos         = arrqmc,
                    corrections=[],
                    calculations=[
                        loop(max=4, qmc=linopt2),
                        loop(max=4, qmc=linopt3),
                    ],
                    dependencies=[
                        (opt_orbitals, "orbitals"),
                        (opt_J2, "jastrow"),
                    ],
                )

                vmc_dmc_dep = opt_J3
            # end if
        # end if

        if run_vmc:
            vmc_path = directory
            if j3:
                vmc_path = directory
            # end if
            vmcrun = generate_qmcpack(
                identifier="vmc",
                path=vmc_path,
                job=qmc_job,
                input_type="basic",
                bconds=boundaries,
                system=nscf_system,
                pseudos=arrqmc,
                spin_polarized=True,
                corrections=[],
                jastrows=[],
                calculations=[
                    vmc(
                        warmupsteps=100,
                        samples=256000,
                        blocks=160,
                        #steps=10,
                        #stepsbetweensamples=1,
                        #walkers=1,
                        timestep=0.3,
                        #substeps=4,
                    )
                ],
                dependencies=[(p2q, "orbitals"), (vmc_dmc_dep, "jastrow")],
            )
        else:
            # run DMC
            nkgrid = len(nscf_system.structure.kpoints)
            dmc_nnodes = max(8, nkgrid)
            dmc_job = job(hours=168,nodes=dmc_nnodes,processes_per_node=1,threads=16,app='-n 8 --mpi=pmi2 /home/dtw2/qmcpack-3.14.0-onepai-2021.1.0/bin/qmcpack_complex', presub=qmcpack_presub)
            dmcrun = generate_qmcpack(
                identifier="dmc",
                path="dmc-twod-12-{}-{}-{}".format(scale, scell_vol, dt_dmc),
                job=dmc_job,
                input_type="basic",
                bconds=boundaries,
                system=nscf_system,
                pseudos         = arrqmc,
                # corrections     = [],
                jastrows=[],
                spin_polarized=True,
                # estimators     = [spindensity(grid=grid_density)],
                corrections=["mpc", "chiesa"],
                calculations=[
                    vmc(
                        warmupsteps=100,
                        blocks=nkgrid * 100,
                        steps=1,
                        stepsbetweensamples=1,
                        walkers=1,
                        timestep=0.3,
                        substeps=4,
                        samplesperthread=nkgrid * 30,
                    ),
                    dmc(
                        warmupsteps=dmc_eqblocks,
                        blocks=int((dmcblocks / (sqrt(dt_dmc)) / 10)),
                        steps=10,
                        timestep=dt_dmc,
                        nonlocalmoves=False, #Add true if you want to use T-moves
                    ),
                ],
                dependencies=[(p2q, "orbitals"), (vmc_dmc_dep, "jastrow")],
            )
            # end if
            #dmcrun["pseudos"] = qmc_pseudos
    # end if
run_project()
