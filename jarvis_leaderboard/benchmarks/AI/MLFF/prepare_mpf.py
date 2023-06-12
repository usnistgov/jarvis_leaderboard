# wget https://figshare.com/ndownloader/files/37587100 -O block_0.p
# wget https://figshare.com/ndownloader/files/37587103 -O block_1.p
import pickle as pk
import pandas as pd
import pymatgen
from alignn.data import get_id_train_val_test
from jarvis.core.atoms import pmg_to_atoms
from jarvis.db.jsonutils import dumpjson

print("loading the MPF dataset 2021")
with open("block_0.p", "rb") as f:
    data = pk.load(f)

with open("block_1.p", "rb") as f:
    data2 = pk.load(f)
print("MPF dataset 2021 loaded")
data.update(data2)
df = pd.DataFrame.from_dict(data)
id_train, id_val, id_test = get_id_train_val_test(
    total_size=len(data),
    split_seed=42,
    train_ratio=0.90,
    val_ratio=0.05,
    test_ratio=0.05,
    keep_data_order=False,
)
dataset_train = []
dataset_val = []
dataset_test = []

cnt = 0
for idx, item in df.items():
    # import pdb; pdb.set_trace()
    if cnt in id_train:
        for iid in range(len(item["energy"])):
            dataset_train.append(
                {
                    "id": item["id"][iid],
                    "atoms": pmg_to_atoms(item["structure"][iid]).to_dict(),
                    "energy": item["energy"][iid] / len(item["force"][iid]),
                    "force": (np.array(item["force"][iid])).tolist(),
                    "stress": (np.array(item["stress"][iid])).tolist(),
                }
            )
    elif cnt in id_val:
        for iid in range(len(item["energy"])):
            dataset_val.append(
                {
                    "id": item["id"][iid],
                    "atoms": pmg_to_atoms(item["structure"][iid]).to_dict(),
                    "energy": item["energy"][iid] / len(item["force"][iid]),
                    "force": np.array(item["force"][iid]).tolist(),
                    "stress": np.array(item["stress"][iid]).tolist(),
                }
            )
    elif cnt in id_test:
        for iid in range(len(item["energy"])):
            dataset_test.append(
                {
                    "id": item["id"][iid],
                    "atoms": pmg_to_atoms(item["structure"][iid]).to_dict(),
                    "energy": item["energy"][iid] / len(item["force"][iid]),
                    "force": np.array(item["force"][iid]).tolist(),
                    "stress": np.array(item["stress"][iid]).tolist(),
                }
            )
    cnt += 1

print(
    "using %d samples to train, %d samples to evaluate, and %d samples to test"
    % (len(dataset_train), len(dataset_val), len(dataset_test))
)
# using 168917 samples to train, 9378 samples to evaluate, and 9389 samples to test
mem = dataset_train + dataset_val + dataset_test
dumpjson(data=mem, filename="m3gnet_mpf.json")
m = {}
train = {}
val = {}
test = {}
for i in dataset_train:
    train[i["id"]] = i["energy"]
for i in dataset_val:
    val[i["id"]] = i["energy"]
for i in dataset_test:
    test[i["id"]] = i["energy"]
m["train"] = train
m["val"] = val
m["test"] = test
dumpjson(data=m, filename="m3gnet_mpf_energy.json")


m = {}
train = {}
val = {}
test = {}
for i in dataset_train:
    train[i["id"]] = ";".join(map(str, np.array(i["force"]).flatten()))
for i in dataset_val:
    val[i["id"]] = ";".join(map(str, np.array(i["force"]).flatten()))
for i in dataset_test:
    test[i["id"]] = ";".join(map(str, np.array(i["force"]).flatten()))
m["train"] = train
m["val"] = val
m["test"] = test
dumpjson(data=m, filename="m3gnet_mpf_force.json")

m = {}
train = {}
val = {}
test = {}
for i in dataset_train:
    train[i["id"]] = ";".join(map(str, np.array(i["stress"]).flatten()))
for i in dataset_val:
    val[i["id"]] = ";".join(map(str, np.array(i["stress"]).flatten()))
for i in dataset_test:
    test[i["id"]] = ";".join(map(str, np.array(i["stress"]).flatten()))
m["train"] = train
m["val"] = val
m["test"] = test
dumpjson(data=m, filename="m3gnet_mpf_stress.json")
