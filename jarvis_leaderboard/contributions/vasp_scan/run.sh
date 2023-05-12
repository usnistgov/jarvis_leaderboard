# List of materials to run high-throughput calculations on


ids = ['POSCAR-1002.vasp','POSCAR-113.vasp','POSCAR-1174.vasp','POSCAR-14606.vasp','POSCAR-14813.vasp','POSCAR-21208.vasp','POSCAR-25065.vasp','POSCAR-890.vasp','POSCAR-984.vasp',
'POSCAR-104.vasp','POSCAR-1145.vasp','POSCAR-1180.vasp','POSCAR-14615.vasp','POSCAR-182.vasp','POSCAR-23862.vasp','POSCAR-25114.vasp','POSCAR-91.vasp','POSCAR-1130.vasp','POSCAR-116.vasp','POSCAR-14604.vasp','POSCAR-14648.vasp','POSCAR-20326.vasp','POSCAR-23864.vasp','POSCAR-25180.vasp','POSCAR-963.vasp','POSCAR-1453.vasp','POSCAR-1408.vasp','POSCAR-1405.vasp','POSCAR-1393.vasp','POSCAR-1327.vasp','POSCAR-1315.vasp','POSCAR-1312.vasp','POSCAR-1300.vasp','POSCAR-1294.vasp','POSCAR-1267.vasp','POSCAR-1216.vasp','POSCAR-1201.vasp','POSCAR-1198.vasp','POSCAR-1192.vasp','POSCAR-1189.vasp','POSCAR-1183.vasp','POSCAR-75.vasp','POSCAR-72.vasp','POSCAR-5.vasp','POSCAR-57.vasp','POSCAR-54.vasp','POSCAR-39.vasp','POSCAR-32.vasp','POSCAR-30.vasp','POSCAR-299.vasp','POSCAR-23.vasp','POSCAR-1954.vasp','POSCAR-17.vasp','POSCAR-1702.vasp','POSCAR-1186-InAs.vasp','POSCAR-96.vasp','POSCAR-95.vasp','POSCAR-9147.vasp','POSCAR-8583.vasp','POSCAR-8566.vasp','POSCAR-8169.vasp','POSCAR-8158.vasp','POSCAR-8082.vasp','POSCAR-8003.vasp','POSCAR-7860.vasp','POSCAR-7844.vasp','POSCAR-7762.vasp','POSCAR-7678.vasp','POSCAR-7630.vasp']

from jarvis.tasks.vasp.vasp import (
    JobFactory,
    VaspJob,
    GenericIncars,
    write_jobfact,
)
from jarvis.io.vasp.inputs import Potcar, Incar, Poscar
from jarvis.db.jsonutils import dumpjson
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from jarvis.tasks.queue_jobs import Queue
import os
vasp_cmd = "/usr/bin/mpirun vasp_std"
#copy_files = ["/home/dtw2/vdw_kernel.bindat"]
#submit_cmd = ["qsub", "submit_job"]
run_dir="/rk3/dtw2/updated-scan-leaderboard"
# For slurm
submit_cmd = ["sbatch", "submit_job"]

steps = [
    "ENCUT",
    "KPLEN",
    "RELAX",
    "BANDSTRUCT",
    "OPTICS",
    "ELASTIC",
]
incs = GenericIncars().scan().incar.to_dict()

for id in ids:
    mat = Poscar.from_file(id)
    cwd_home = os.getcwd()
    dir_name = id.split('.vasp')[0] + "_" + str("PBEBO")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    os.chdir(dir_name)
    job = JobFactory(
        vasp_cmd=vasp_cmd,
        poscar=mat,
        steps=steps,
        #copy_files=copy_files,
        use_incar_dict=incs,
    )

    dumpjson(data=job.to_dict(), filename="job.json")
    write_jobfact(
        pyname="job.py",
        job_json="job.json",
        input_arg="v.step_flow()",
    )

    # Example job commands, need to change based on your cluster
    job_line = "source ~/miniconda3/envs/my_jarvis/bin/activate my_jarvis \npython job.py \n"
    name = id
    directory = os.getcwd()
    Queue.slurm(
    job_line=job_line,
 #   filename=tmp_name,
    nnodes=1,  # rtx
    cores=16,  # rtx
    directory=directory,
    submit_cmd=submit_cmd,
    queue="main",
    walltime="300:00:00"
)
    os.chdir(cwd_home)
