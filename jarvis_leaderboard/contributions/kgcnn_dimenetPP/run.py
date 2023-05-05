# https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/kgcnn_jarvis_leaderboard.ipynb
from kgcnn.literature.DimeNetPP import make_crystal_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
from jarvis.core.atoms import Atoms
from kgcnn.data.crystal import CrystalDataset
import numpy as np
from sklearn.metrics import mean_absolute_error
from jarvis.db.jsonutils import loadjson, dumpjson
from kgcnn.training.hyper import HyperParameter
from kgcnn.model.utils import get_model_class
import glob
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_config =   {
                "name": "DimeNetPP",
                "inputs": [{"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                           {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                           {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                           {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True},
                           {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                           {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
                           ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128,
                                             "embeddings_initializer": {"class_name": "RandomUniform",
                                                                        "config": {"minval": -1.7320508075688772,
                                                                                   "maxval": 1.7320508075688772}}}},
                "emb_size": 128, "out_emb_size": 256, "int_emb_size": 64, "basis_emb_size": 8,
                "num_blocks": 4, "num_spherical": 7, "num_radial": 6,
                "cutoff": 5.0, "envelope_exponent": 5,
                "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
                "num_targets": 1, "extensive": False, "output_init": "zeros",
                "activation": "swish", "verbose": 10,
                "output_embedding": "graph",
                "use_output_mlp": False,
                "output_mlp": {},
            }


# GPU doesnt seem to work with kgcnn
tasks = []
for i in glob.glob("../*/AI-SinglePropertyPrediction*test-mae.csv.zip"):
    if "formula" not in i:
        task = i.split("/")[-1].split(".csv.zip")[0]
        if task not in tasks:
            tasks.append(task)
print("tasks", tasks, len(tasks))
# For a quick test running on one task only
#tasks = ["AI-SinglePropertyPrediction-exfoliation_energy-dft_3d-test-mae"]
for task in tasks:
 t1=time.time()
 zip_name=task+'.csv.zip'
 if not os.path.exists(zip_name):
 #if os.path.exists(zip_name):
    #task = "AI-SinglePropertyPrediction-exfoliation_energy-dft_3d-test-mae"
    cmd = (
        "jarvis_populate_data.py --benchmark_file "
        + task
        + " --output_path=Out"
    )
    if not os.path.exists("Out"):
        os.system(cmd)
    dataset_info = loadjson("Out/dataset_info.json")
    n_train = dataset_info["n_train"]
    n_val = dataset_info["n_val"]
    n_test = dataset_info["n_test"]

    model = make_crystal_model(**model_config)

    model.compile(
        loss="mean_absolute_error",
        optimizer=Adam(learning_rate=1e-04),
        metrics=["mean_absolute_error"],
    )

    def prepare_data(dirname="exfoliation_en", populated_data_path="Out"):
        id_prop_path = os.path.join(populated_data_path, "id_prop.csv")
        df = pd.read_csv(id_prop_path, header=None)
        df.columns = ["id", "target"]

        # train
        train_dir = dirname + "_train"
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        train_cif = os.path.join(train_dir, "CifFiles")
        if not os.path.exists(train_cif):
            os.makedirs(train_cif)
        csv_file = train_dir + "/data.csv"
        print("train_dir", train_dir)
        f = open(csv_file, "w")
        f.write("file_name,index,label\n")
        targets = []
        for i, ii in df.iterrows():
            pos_path = os.path.join(populated_data_path, ii["id"])
            atoms = Atoms.from_poscar(pos_path)
            pmg = atoms.pymatgen_converter()
            targets.append(ii["target"])
            fname = "file_" + str(ii["id"]) + ".cif"
            # fname="file_"+str(i)+".cif"
            cif_name = os.path.join(train_cif, fname)
            pmg.to(filename=cif_name, fmt="cif")
            line = fname + "," + str(i) + "," + str(ii["target"]) + "\n"
            f.write(line)
        f.close()
        dataset = CrystalDataset(
            data_directory=train_dir,
            dataset_name=train_dir,
            file_name="data.csv",
            file_directory="CifFiles",
        )
        dataset.prepare_data(file_column_name="file_name", overwrite=True)
        dataset.read_in_memory(label_column_name="label")
        labels = np.expand_dims(dataset.get("graph_labels"), axis=-1)
        dataset.map_list(
            method="set_range_periodic", max_distance=8.0, max_neighbours=20
        )
        dataset.map_list(method="set_angle", allow_multi_edges=True)
        return dataset, labels, df, train_dir

    dataset, labels, df, train_dir = prepare_data(populated_data_path="Out")

    # n_train=650
    # n_val=81
    # n_test=81

    print("n_train", n_train)
    print("n_val", n_val)
    print("n_test", n_test)

    train_index = np.arange(0, n_train)
    val_index = np.arange(n_train, n_train + n_val)
    test_index = np.arange(n_train + n_val, n_train + n_val + n_test)
    # x_train,y_train=dataset[train_index], labels[train_index]
    # x_train,y_train=dataset[train_index].tensor(), labels[train_index]
    x_train, y_train = (
        dataset[train_index].tensor(model_config["inputs"]),
        labels[train_index],
    )
    x_val, y_val = dataset[val_index], labels[val_index]
    # x_val, y_val = dataset[val_index].tensor(), labels[val_index]
    x_val, y_val = (
        dataset[val_index].tensor(model_config["inputs"]),
        labels[val_index],
    )
    x_test, y_test = dataset[test_index], labels[test_index]
    # x_test, y_test = dataset[test_index].tensor(), labels[test_index]
    x_test, y_test = (
        dataset[test_index].tensor(model_config["inputs"]),
        labels[test_index],
    )
    model.fit(
        x_train,
        y_train,
        shuffle=False,
        batch_size=16,
        epochs=200,
        verbose=2,
    )
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    print(
        mean_absolute_error(y_val, val_pred),
        mean_absolute_error(y_test, test_pred),
    )
    df_test = df[-n_test:]
    csv_name = task + ".csv"
    f = open(csv_name, "w")
    f.write("id,prediction\n")
    for i in range(len(df_test)):
        # print (i)
        jid = df_test.iloc[i]["id"]
        target = df_test.iloc[i]["target"]
        # print(jid,target,y_test[i][0],test_pred[i][0])
        line = jid + "," + str(test_pred[i][0]) + "\n"
        f.write(line)
    f.close()

    cmd = "zip " + csv_name + ".zip " + csv_name
    os.system(cmd)

    cmd = "rm -r Out"
    os.system(cmd)
    cmd = "rm -r exfoliation_en_train"
    os.system(cmd)
    cmd = "rm " + csv_name
    #os.system(cmd)
    cmd = "rm -r " + train_dir
    os.system(cmd)
 t2=time.time()
 print('Time',t2-t1)
