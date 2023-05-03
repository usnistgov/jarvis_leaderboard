"""Module to prepare benchmark dataset from id list."""
from jarvis.db.figshare import data
from jarvis.db.jsonutils import loadjson, dumpjson
import pandas as pd
from collections import defaultdict

# The ids_train_val_test.json files are
# obtained from Figshare:
# https://figshare.com/projects/ALIGNN_models/126478
def preapre_json_file(
    id_tag="id",
    dataset="hmof",
    prop="exfoliation_energy",
    train_val_test="ids_train_val_test.json",
):
    """Prepare json file given id,dataset and prop info."""
    print("Usind ids_train_val_test.json", train_val_test)
    if prop=='gappbe':
       prop='gap pbe'
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
    print ('type',type(train_ids[0]))
    train_data = defaultdict()
    for i,j in df[df[id_tag].isin(train_ids)][[id_tag,prop]].iterrows():
         #if dataset=='hmof':
         #  train_data['hMOF-'+str(j[id_tag])]=j[prop]
         #  #train_data['hMOF-'+str(i)]=j[prop]
         #else:
           train_data[str(j[id_tag])]=j[prop]
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
         #if dataset=='hmof':
         #  val_data['hMOF-'+str(j[id_tag])]=j[prop]
         #  #val_data['hMOF-'+str(i)]=j[prop]
         #else:
           val_data[str(j[id_tag])]=j[prop]



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
         #if dataset=='hmof':
         #  test_data['hMOF-'+str(j[id_tag])]=j[prop]
         #  #test_data['hMOF-'+str(i)]=j[prop]
         #else:
           test_data[str(j[id_tag])]=j[prop]

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
    fname = dataset + "_" + prop + ".json"
    dumpjson(data=mem, filename=fname)
    print(
        "train_ids,val_ids,test_ids",
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )
    return mem


    # print('get_val',len(train_data))


if __name__ == "__main__":
    preapre_json_file(prop="formation_energy_peratom")
