"""Module to prepare benchmark dataset from id list."""
from jarvis.db.figshare import data
from jarvis.db.jsonutils import loadjson, dumpjson
import pandas as pd
from collections import defaultdict

# The ids_train_val_test.json files are
# obtained from Figshare:
# https://figshare.com/projects/ALIGNN_models/126478
def preapre_json_file(
    id_tag="jid",
    dataset="dft_3d",
    prop="exfoliation_energy",
    train_val_test="ids_train_val_test.json",
):
    """Prepare json file given id,dataset and prop info."""
    print("Usind ids_train_val_test.json", train_val_test)
    d = data(dataset)
    df = pd.DataFrame(d)

    def get_val(id_tag=id_tag, prop=prop, jv_id="JVASP-14441"):
        """Get data from dataframe."""
        # print (df[df[id_tag]=='JVASP-14441'])
        # return df[df[id_tag]==id][prop].values[0]
        return df[df[id_tag] == jv_id][prop].values[0]

    split_file = loadjson(train_val_test)
    train_ids = split_file["id_train"]
    val_ids = split_file["id_val"]
    test_ids = split_file["id_test"]

    train_data = defaultdict()
    for i in train_ids:
        val = get_val(jv_id=i)
        if val == "na":
            print(i, val)
            import sys

            sys.exit()
        train_data[i] = val

    val_data = defaultdict()
    for i in val_ids:
        val = get_val(jv_id=i)
        if val == "na":
            print(i, val)
            import sys

            sys.exit()
        val_data[i] = val
    test_data = defaultdict()
    for i in test_ids:
        val = get_val(jv_id=i)
        if val == "na":
            print(i, val)
            import sys

            sys.exit()
        test_data[i] = val

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
