from jarvis.db.jsonutils import loadjson
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import get_request_data
import os,json, zipfile,time
#From https://doi.org/10.6084/m9.figshare.23290220
#d = loadjson("halide_peroskites.json")
#train_list_ids provided by authors
#f = open("train_list_ids", "r")
#lines = f.read().splitlines()
#f.close()
#train_ids = []
#for i in lines:
#    id = "Perov-" + str(i.split()[0])
#    train_ids.append(id)

#m = {}
#train = {}
#test = {}
#for i in d:
#    if i["id"] in train_ids:
#        train[i["id"]] = i["PBE_gap"]
#    else:
#        test[i["id"]] = i["PBE_gap"]

#m["train"] = train
#m["test"] = test
#dumpjson(data=m, filename="halide_peroskites_PBE_gap.json")

#d = loadjson("halide_peroskites.json")
d = get_request_data(js_tag="halide_peroskites.json",url="https://figshare.com/ndownloader/files/41046323")
prop="PBE_gap"
fname="../../../benchmarks/AI/SinglePropertyPrediction/halide_peroskites_"+prop+".json.zip"
tag = "halide_peroskites_"+prop+".json"
d2 = json.loads(zipfile.ZipFile(fname).read(tag)) #loadjson("halide_peroskites_PBE_gap.json")
train_ids = d2["train"]
test_ids = d2["test"]
dir_name = "DataDir_"+prop
if not os.path.exists(dir_name):
 os.makedirs(dir_name)
fname="DataDir_"+prop+"/id_prop.csv"
f = open(fname, "w")
for i in d:
    if i["id"] in train_ids:
        atoms = Atoms.from_dict(i["atoms"])
        fname = "DataDir_"+prop+"/" + i["id"]
        atoms.write_poscar(fname)
        prop_val = i[prop]
        line = i["id"] + "," + str(prop_val) + "\n"
        f.write(line)
print(len(train_ids))
for i in d:
    if i["id"] in test_ids:
        atoms = Atoms.from_dict(i["atoms"])
        fname = "DataDir_"+prop+"/" + i["id"]
        atoms.write_poscar(fname)
        prop_val = i[prop]
        line = i["id"] + "," + str(prop_val) + "\n"
        f.write(line)
for i in d:
    if i["id"] in test_ids:
        atoms = Atoms.from_dict(i["atoms"])
        fname = "DataDir_"+prop+"/" + i["id"]
        atoms.write_poscar(fname)
        prop_val = i[prop]
        line = i["id"] + "," + str(prop_val) + "\n"
        f.write(line)
f.close()
print(len(test_ids))
t1=time.time()
cmd = 'train_folder.py --root_dir DataDir_'+prop+'/ --config_name config_example_halide.json  --output_dir="Out_'+prop+'"'
os.system(cmd)
cmd='DataDir_'+prop+'/prediction_results_test_set.csv '+'AI-SinglePropertyPrediction-'+prop+'-halide_peroskites-test-mae.csv'
os.system(cmd)
cmd='AI-SinglePropertyPrediction-'+prop+'-halide_peroskites-test-mae.csv.zip '+'AI-SinglePropertyPrediction-'+prop+'-halide_peroskites-test-mae.csv'
t2=time.time()
print('Time',t2-t1)
