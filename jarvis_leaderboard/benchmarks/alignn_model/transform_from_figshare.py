"""Module to prepare benchmark dataset from id list."""
from jarvis.db.figshare import data
from jarvis.db.jsonutils import loadjson, dumpjson
import pandas as pd
from collections import defaultdict
import glob, os
import zipfile, json
from jarvis.db.jsonutils import dumpjson
#from format_data import preapre_json_file
import pandas as pd
from io import StringIO

# The ids_train_val_test.json files are
# obtained from Figshare:
# https://figshare.com/projects/ALIGNN_models/126478
def preapre_json_file(
    id_tag="id",
    dataset="hmof",
    prop="exfoliation_energy",
    train_val_test="ids_train_val_test.json",
    class_val=None,
):
    """Prepare json file given id,dataset and prop info."""
    print("Usind ids_train_val_test.json", train_val_test)
    #if dataset=='qm9':
    #   dataset='qm9_std_jctc'
    d = data(dataset)
    df = pd.DataFrame(d,dtype='str')
   
    def get_val(id_tag=id_tag, prop=prop, jv_id="JVASP-14441"):
        """Get data from dataframe."""
        # print (df[df[id_tag]=='JVASP-14441'])
        # return df[df[id_tag]==id][prop].values[0]
        return df[df[id_tag] == jv_id][prop].values[0]

    split_file = loadjson(train_val_test)
    train_ids = split_file["id_train"]
    val_ids = split_file["id_val"]
    test_ids = split_file["id_test"]
    #print ('type',type(train_ids[0]))
    train_data = defaultdict()
    for i,j in df[df[id_tag].isin(train_ids)][[id_tag,prop]].iterrows():
         if dataset=='hmof':
           train_data['hMOF-'+str(j[id_tag])]=j[prop]
           if class_val is not None:
                 if float(j[prop])<=class_val:
                    train_data['hMOF-'+str(j[id_tag])]=0
                 else:
                    train_data['hMOF-'+str(j[id_tag])]=1
                    
           #train_data['hMOF-'+str(i)]=j[prop]
         else:
           train_data[str(j[id_tag])]=j[prop]
           if class_val is not None:
                 #print('jprop',j[prop],type(j[prop]))
                 if float(j[prop])<=class_val:
                    train_data[str(j[id_tag])]=0
                 else:
                    train_data[str(j[id_tag])]=1
    #print ('i',i)
    #print('j',j)
    #print('train')
    #for i in train_ids:
    #    val = get_val(jv_id=i)
    #    if val == "na":
    #        print(i, val)
    #        import sys

    #        sys.exit()
    #    train_data[i] = val

    val_data = defaultdict()
    for i,j in df[df[id_tag].isin(val_ids)][[id_tag,prop]].iterrows():
         #val_data[i]=j[prop]
         if dataset=='hmof':
           val_data['hMOF-'+str(j[id_tag])]=j[prop]
           if class_val is not None:
                 #if j[prop]<=class_val:
                 if float(j[prop])<=class_val:
                    val_data['hMOF-'+str(j[id_tag])]=0
                 else:
                    val_data['hMOF-'+str(j[id_tag])]=1
           #val_data['hMOF-'+str(i)]=j[prop]
         else:
           val_data[str(j[id_tag])]=j[prop]
           if class_val is not None:
                 #if j[prop]<=class_val:
                 if float(j[prop])<=class_val:
                    val_data[str(j[id_tag])]=0
                 else:
                    val_data[str(j[id_tag])]=1



    #print('val')
    #for i in val_ids:
    #    val = get_val(jv_id=i)
    #    if val == "na":
    #        print(i, val)
    #        import sys

    #        sys.exit()
    #    val_data[i] = val
    test_data = defaultdict()
    for i,j in df[df[id_tag].isin(test_ids)][[id_tag,prop]].iterrows():
         #test_data[i]=j[prop]
         if dataset=='hmof':
           test_data['hMOF-'+str(j[id_tag])]=j[prop]
           if class_val is not None:
                 #if j[prop]<=class_val:
                 if float(j[prop])<=class_val:
                    test_data['hMOF-'+str(j[id_tag])]=0
                 else:
                    test_data['hMOF-'+str(j[id_tag])]=1
           #test_data['hMOF-'+str(i)]=j[prop]
         else:
           test_data[str(j[id_tag])]=j[prop]
           if class_val is not None:
                 #if j[prop]<=class_val:
                 if float(j[prop])<=class_val:
                    test_data[str(j[id_tag])]=0
                 else:
                    test_data[str(j[id_tag])]=1

    print('test',len(test_ids),len(test_data))
   # for i in test_ids:  
   #   print('dddd',test_data[i])
    #for i in test_ids:
    #    val = get_val(jv_id=i)
    #    if val == "na":
    #        print(i, val)
    #        import sys

    #        sys.exit()
    #    test_data[i] = val

    mem = {}
    mem["train"] = dict(train_data)
    mem["val"] = dict(val_data)
    mem["test"] = dict(test_data)
    fname = dataset + "_" + prop.replace("n-", "n_").replace("p-", "p_") + ".json"
    dumpjson(data=mem, filename=fname)
    print(
        "train_ids,val_ids,test_ids",
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )
    return mem

    # print('get_val',len(train_data))


#if __name__ == "__main__":
#    preapre_json_file(prop="formation_energy_peratom")
pp = list(
    set(
        [
            "avg_elec_mass",
            "avg_hole_mass",
            "bulk_modulus_kv",
            "dfpt_piezo_max_dielectric",
            "dfpt_piezo_max_dij",
            "ehull",
            "encut",
            "epsx",
            "epsy",
            "epsz",
            "exfoliation_energy",
            "formation_energy_peratom",
            "kpoint_length_unit",
            "magmom_oszicar",
            "magmom_oszicar",
            "max_efg",
            "mbj_bandgap",
            "mbj_bandgap",
            "mepsx",
            "mepsy",
            "mepsz",
            "n-powerfact",
            "n-powerfact",
            "n-Seebeck",
            "optb88vdw_bandgap",
            "optb88vdw_bandgap",
            "optb88vdw_total_energy",
            "shear_modulus_gv",
            "slme",
            "spillage",
        ]
    )
)
pp = list(
    set(
        [
            "lcd",
            "pld",
            "void_fraction",
            "surface_area_m2g",
            "surface_area_m2cm3",
            "max_co2_adsp",
            "min_co2_adsp",
        ]
    )
)
# pp=['slme','spillage','shear_modulus_gv']



pp = ['alpha','G','Cv','gap','H','HOMO','LUMO','mu','R2','U','ZPVE','U0']
class_val=None
#pp = ['alpha']#,'G','Cv','gap','H','HOMO','LUMO','mu','R2','U','ZPVE','U0']
dataset='qm9_std_jctc'
id_tag='id'

thresholds = {
    "optb88vdw_bandgap": 0.01,
    "mbj_bandgap": 0.01,
    "slme": 10,
    "ehull": 0.1,
    "magmom_oszicar": 0.05,
    "spillage": 0.1,
    "n-Seebeck": -100,
    "p-Seebeck": 100,
    "n-powerfact": 1000,
    "p-powerfact": 1000,
}

pp=['optb88vdw_bandgap','magmom_oszicar','mbj_bandgap','spillage','slme','p-Seebeck','n-powerfact']
pp=['Band_gap_HSE']
dataset='dft_3d'
dataset='snumat'
id_tag='jid'
id_tag='id'
id_tag="SNUMAT_id"
class_val=None
x = []

for i in glob.glob(
    #"../../../JARVIS-ALIGNN/Models/qm9_std_jctc/*.zip"
    #"../../../JARVIS-ALIGNN/Models/17005906/*.zip"
    #"../../../JARVIS-ALIGNN/Models/17005987/*.zip"
    #"../../../JARVIS-ALIGNN/Models/17005681/*.zip"
    "../../../JARVIS-ALIGNN/Models/snumat/*.zip"
):
    if "supercon" not in i:
        p = (
            i.split("/")[-1]
            .split(".zip")[0]
            .split("snumat_")[1]
            #.split("jv_")[1]
            #.split("qm9_std_jctc_")[1]
            #.split("hmof_")[1]
            #.split("_alignn_class")[0]
            .split("_alignn")[0]
            #.split("_alignnn")[0]
        )
        if p in pp:
            #class_val=thresholds[p]
            #model_zipfile = (
            #    #"../../../JARVIS-ALIGNN/Models/qm9_std_jctc/qm9_std_jctc_"
            #    #"../../../JARVIS-ALIGNN/Models/17005906/qm9_"
            #    #"../../../JARVIS-ALIGNN/Models/17005987/hmof_"
            #    #"../../../JARVIS-ALIGNN/Models/17005681/hmof_"
            #    "../../../JARVIS-ALIGNN/Models/17005681/jv_"
            #    + p
            #    + "_alignn.zip"
            #    #+ "_alignnn.zip"
            #)
            #model_zip = zipfile.ZipFile(model_zipfile)
            model_zip = zipfile.ZipFile(i)
            print("model_zipfile", i, model_zip.namelist())
            #temp = "qm9_std_jctc_" + p + "_alignn/ids_train_val_test.json"
            #temp = "qm9_" + p + "_alignn/ids_train_val_test.json"
            #temp = "hmof_" + p + "_alignnn/ids_train_val_test.json"
            #temp = "jv_" + p + "_alignn/ids_train_val_test.json"
            #temp = "jv_" + p + "_alignn_class/ids_train_val_test.json"
            temp = "snumat_" + p + "_alignn/ids_train_val_test.json"
            train_val_test = json.loads(model_zip.read(temp))

            dumpjson(filename="ids_train_val_test.json", data=train_val_test)
            preapre_json_file(prop=p,dataset=dataset,id_tag=id_tag,class_val=class_val)
            cmd = "rm ids_train_val_test.json"
            #os.system(cmd)

            fname = p.replace("n-", "n_").replace("p-", "p_") + ".md"
            f = open(fname, "w")
            line = "# Model for " + p + "\n\n"
            f.write(line)
            line = '<h2>Model benchmarks</h2>\n\n<table style="width:100%" id="j_table">\n <thead>\n  <tr>\n    <th>Model name</th>\n<th>Dataset</th>\n   <!-- <th>Method</th>-->\n    <th>ACC</th>\n    <th>Team name</th>\n    <th>Dataset size</th>\n    <th>Date submitted</th>\n    <th>Notes</th>\n  </tr>\n </thead>\n<!--table_content-->\n</table>\n'
            f.write(line)
            f.close()
            temp = "snumat_" + p + "_alignn/prediction_results_test_set.csv"
            #temp = "jv_" + p + "_alignn_class/prediction_results_test_set.csv"
            #temp = "qm9_std_jctc_" + p + "_alignn/prediction_results_test_set.csv"
            #temp = "hmof_" + p + "_alignnn/prediction_results_test_set.csv"
            p = p.replace("n-", "n_").replace("p-", "p_")  # For n-seebeck etc
            #fname = "SinglePropertyPrediction-test-" + p + "-qm9_std_jctc-AI-mae.csv"
            #fname = "SinglePropertyPrediction-test-" + p + "-hmof-AI-mae.csv"
            #fname = "SinglePropertyClass-test-" + p + "-dft_3d-AI-acc.csv"
            #fname = "SinglePropertyPrediction-test-" + p + "-dft_3d-AI-acc.csv"
            fname = "SinglePropertyPrediction-test-" + p + "-snumat-AI-mae.csv"
            f = open(fname, "wb")
            f.write(model_zip.read(temp))
            f.close()
            # df=pd.read_csv(StringIO(model_zip.read(temp)))
            # print (df)
            # print(i,len(train_val_test))
            x.append(p)
print(x)

import os, glob

for i in glob.glob("*csv"):
    cmd = "zip " + i + ".zip " + i
    os.system(cmd)
for i in glob.glob("*.json"):
    cmd = "zip " + i + ".zip " + i
    os.system(cmd)
