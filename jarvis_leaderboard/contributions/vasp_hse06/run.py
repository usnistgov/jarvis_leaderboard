from jarvis.tasks.vasp.vasp import GenericIncars
from jarvis.db.jsonutils import dumpjson
from jarvis.core.atoms import Atoms
from jarvis.tasks.vasp.vasp import VaspJob, write_vaspjob
from jarvis.tasks.queue_jobs import Queue
from jarvis.io.vasp.inputs import Poscar
from jarvis.db.figshare import data
from jarvis.io.vasp.inputs import Poscar, Incar, Potcar
from jarvis.core.kpoints import Kpoints3D

inc = Incar(data)


def get_symb(atoms):
    new_symb = []
    for i in atoms.elements:
        if i not in new_symb:
            new_symb.append(i)
    return new_symb


dat = data("dft_3d")

incs = GenericIncars().hse06().incar.to_dict()
vasp_cmd = "ibrun vasp_std"
copy_files = [""]
jids_prior = []
ids = ["JVASP-1002"]
for i in dat:

    if i["jid"] in jids:
        strt = Atoms.from_dict(i["atoms"])
        encut = i["encut"]
        name = i["jid"] + "_HSE06"
        print("dir_name", name)
        if not os.path.exists(name):
            os.makedirs(name)
        os.chdir(name)
        pos = Poscar(strt)
        symb = get_symb(strt)
        inc.update({"ENCUT": i["encut"]})
        pos.comment = name
        pot = Potcar(elements=symb)
        # print (pos)
        # print (pos.atoms.elements)
        # print ()
        # print ('symb',symb)
        kp = (
            Kpoints3D()
            .automatic_length_mesh(
                lattice_mat=strt.lattice_mat, length=i["kpoint_length_unit"]
            )
            ._kpoints
        )
        # kp[0][2]=1
        kpoints = Kpoints3D(kpoints=kp)

        jobname = name

        job = VaspJob(
            poscar=pos,
            incar=inc,
            potcar=pot,
            kpoints=kpoints,
            vasp_cmd=vasp_cmd,
            copy_files=copy_files,
            jobname=jobname,
        )
        dumpjson(data=job.to_dict(), filename="job.json")
        write_vaspjob(pyname="job.py", job_json="job.json")
        submit_cmd = ["sbatch", "submit_job"]
        job_line = "\nmodule load vasp/6.3.0\n \nexport VASP_PSP_DIR=PATH TO VASP-POTENTIAL\n \npython job.py \n"
        directory = os.getcwd()

        Queue.slurm(
            job_line=job_line,
            jobname=name,
            directory=directory,
            queue="small",
            walltime="48:00:00",
            submit_cmd=submit_cmd,
        )
        os.chdir("..")
