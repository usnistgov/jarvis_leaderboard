from matformer.train_props import train_prop_model 
import os
import time
#Added ids https://github.com/YKQ98/Matformer/blob/main/matformer/train.py#L578
props = [
    #"exfoliation_energy",
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "mbj_bandgap",
    "slme",
    "magmom_oszicar",
    "spillage",
    "kpoint_length_unit",
    "encut",
    "optb88vdw_total_energy",
    "epsx",
    "epsy",
    "epsz",
    "mepsx",
    "mepsy",
    "mepsz",
    "max_ir_mode",
    "min_ir_mode",
    "n-Seebeck",
    "p-Seebeck",
    "n-powerfact",
    "p-powerfact",
    "ncond",
    "pcond",
    "nkappa",
    "pkappa",
    "ehull",
    "dfpt_piezo_max_dielectric",
    "dfpt_piezo_max_eij",
    "dfpt_piezo_max_dij",
]
#train_prop_model(learning_rate=0.001,test_only=False,name="matformer",num_workers=0, prop=props[0], pyg_input=True, n_epochs=2, batch_size=32, use_lattice=True, output_dir="./matformer_jarvis_formation_energy", use_angle=False, save_dataloader=False)

for i in props:
   output_dir = 'out_'+i
   csv_path=output_dir+'/prediction_results_test_set.csv'
   if not os.path.exists(csv_path):
        t1=time.time()
        train_prop_model(learning_rate=0.001,name="matformer", prop=i, pyg_input=True, n_epochs=50, batch_size=64, use_lattice=True, output_dir=output_dir, use_angle=False, save_dataloader=False)
        t2=time.time()
        print ('TIME for training',t2-t1)
