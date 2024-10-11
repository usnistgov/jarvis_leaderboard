from scipy.stats import pearsonr, spearmanr
from jarvis.db.figshare import data
from jarvis.db.jsonutils import loadjson, dumpjson
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph
import torch, os
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
import pandas as pd

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def load_model(
    dir_path="", config_filename="config.json", filename="best_model.pt"
):
    best_path = os.path.join(dir_path, filename)
    config = loadjson(os.path.join(dir_path, config_filename))
    # print("config",config)
    model = ALIGNNAtomWise(ALIGNNAtomWiseConfig(**config["model"]))
    model.state_dict()
    model.load_state_dict(
        torch.load(os.path.join(dir_path, filename), map_location=device)
    )
    model.to(device)
    model.eval()
    return model


def evaluate(
    dir_path="",
    csv_file="",
    filename="current_model.pt",
    config_filename="config.json",
):
    # dir_path='temp/'
    # csv_file='temp/prediction_results_test_set.csv'
    # config_filename='config.json'
    model = load_model(dir_path=dir_path, filename=filename)
    df = pd.read_csv(csv_file)
    orig = pearsonr(df["target"], df["prediction"])[0]
    tmp = csv_file.split("/")[0].split("-")
    # print("tmp", tmp)
    dataset = tmp[4]
    prop = tmp[3]
    dat = data(dataset)
    config = loadjson(os.path.join(dir_path, config_filename))
    if "jid" in dat[0]:
        id_tag = "jid"
    else:
        id_tag = "id"
    targ = []
    pred = []
    # print('keys',dat[0].keys())
    # print('df',csv_file,df)
    for ii in dat:
        # if ii[id_tag] in test:
        if ii[id_tag] in list(df["id"].values):
            # nm="DataDir/"+ii[id_tag]
            atoms = Atoms.from_dict(ii["atoms"])
            g, lg = Graph.atom_dgl_multigraph(
                atoms,
                neighbor_strategy=config["neighbor_strategy"],
                cutoff=config["cutoff"],
                max_neighbors=config["max_neighbors"],
                atom_features=config["atom_features"],
                use_canonize=config["use_canonize"],
            )
            result = (
                (model((g.to(device), lg.to(device))))["out"]
                .cpu()
                .detach()
                .numpy()
            )
            # print("result", result, ii[prop])
            targ.append(ii[prop])
            pred.append(result)
    current = pearsonr(targ, pred)[0]
    # print("orig,pred", orig, current, max(orig, current))
    return current  # max(orig, current)


bench_files = [
    "AI-SinglePropertyPrediction-ead-AGRA_CHO-test-mae.csv.zip",
    "AI-SinglePropertyPrediction-ead-AGRA_COOH-test-mae.csv.zip",
    "AI-SinglePropertyPrediction-ead-AGRA_CO-test-mae.csv.zip",
    "AI-SinglePropertyPrediction-ead-AGRA_O-test-mae.csv.zip",
    "AI-SinglePropertyPrediction-ead-AGRA_OH-test-mae.csv.zip",
    "AI-SinglePropertyPrediction-ead-tinnet_N-test-mae.csv.zip",
    "AI-SinglePropertyPrediction-ead-tinnet_O-test-mae.csv.zip",
    "AI-SinglePropertyPrediction-ead-tinnet_OH-test-mae.csv.zip",
]
# wget https://raw.githubusercontent.com/usnistgov/alignn/main/alignn/examples/sample_data/config_example.json
config = loadjson("config_example.json")
for i in bench_files:
    out_dir = "Out-" + i
    data_dir = "DataDir-" + i
    config_name = "config-" + i
    cmd = "rm -r sample*"
    os.system(cmd)
    csv_file = os.path.join(out_dir, "prediction_results_test_set.csv")
    if not os.path.exists(csv_file):
        cmd = (
            "jarvis_populate_data.py --benchmark_file "
            + i
            + " --output_path "
            + data_dir
            + " --id_tag id"
        )
        print(cmd)
        os.system(cmd)
        dataset_info = loadjson(os.path.join(data_dir, "dataset_info.json"))
        n_train = dataset_info["n_train"]
        n_val = dataset_info["n_val"]
        n_test = dataset_info["n_test"]
        config["n_train"] = n_train
        config["n_val"] = n_val
        config["n_test"] = n_test
        config["epochs"] = 300
        config["batch_size"] = 5
        dumpjson(data=config, filename=config_name)

        cmd = (
            "train_alignn.py --root_dir "
            + data_dir
            + " --config "
            + config_name
            + " --output_dir="
            + out_dir
        )
        os.system(cmd)
for i in bench_files:
    out_dir = "Out-" + i
    for j in bench_files:
        out_dirj = "Out-" + j
        csv_file = os.path.join(out_dirj, "prediction_results_test_set.csv")
        pear_best = evaluate(
            dir_path=out_dir, csv_file=csv_file, filename="best_model.pt"
        )
        pear_current = evaluate(
            dir_path=out_dir, csv_file=csv_file, filename="current_model.pt"
        )
        pear = max(pear_best, pear_current)
        print(out_dir, out_dirj, pear)
