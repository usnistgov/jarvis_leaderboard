"""Module to run matminer results."""
import random
import os
import pandas as pd
from tqdm import tqdm
import csv
import numpy as np
import math
import lightgbm as lgb
from jarvis.ai.pkgs.utils import regr_scores
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
import zipfile
import json
import time

tqdm.pandas()

# https://github.com/mathsphy/paper-ml-robustness-material-property/blob/main/myfunc.py
def StructureFeaturizer(
    df_in, col_id="structure", ignore_errors=True, chunksize=30
):
    """
    Featurize a dataframe using Matminter Structure featurizer

    Parameters
    ----------
    df : Pandas.DataFrame
        DataFrame with a column named "structure"

    Returns
    -------
    A DataFrame containing 273 features (columns)

    """
    # For featurization
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (
        ElementProperty,
        Stoichiometry,
        ValenceOrbital,
        IonProperty,
    )
    from matminer.featurizers.structure import (
        SiteStatsFingerprint,
        StructuralHeterogeneity,
        ChemicalOrdering,
        StructureComposition,
        MaximumPackingEfficiency,
    )

    if isinstance(df_in, pd.Series):
        df = df_in.to_frame()
    else:
        df = df_in
    # df[col_id] = df[col_id].apply(to_unitcell)

    # 128 structural feature
    struc_feat = [
        SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"),
        SiteStatsFingerprint.from_preset(
            "LocalPropertyDifference_ward-prb-2017"
        ),
        StructuralHeterogeneity(),
        MaximumPackingEfficiency(),
        ChemicalOrdering(),
    ]
    # 145 compositional features
    compo_feat = [
        StructureComposition(Stoichiometry()),
        StructureComposition(ElementProperty.from_preset("magpie")),
        StructureComposition(ValenceOrbital(props=["frac"])),
        StructureComposition(IonProperty(fast=True)),
    ]
    featurizer = MultipleFeaturizer(struc_feat + compo_feat)
    # Set the chunksize used for Pool.map parallelisation
    featurizer.set_chunksize(chunksize=chunksize)
    featurizer.fit(df[col_id])
    X = featurizer.featurize_dataframe(
        df=df, col_id=col_id, ignore_errors=ignore_errors
    )
    # check failed entries
    print("Featurization completed.")
    failed = np.any(pd.isnull(X.iloc[:, df.shape[1] :]), axis=1)
    if np.sum(failed) > 0:
        print(f"Number failed: {np.sum(failed)}/{len(failed)}")
    return X, failed


props = [
    "exfoliation_energy",
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
    "avg_elec_mass",
    "avg_hole_mass",
    "max_efg",
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
# props = ["exfoliation_energy"]


def load_dataset(
    name: str = "cfid_3d",
    root_dir="",
    file_format="poscar",
    target=None,
    n_train=None,
    n_val=None,
    n_test=None,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    split_seed=123,
):
    id_prop_dat = os.path.join(root_dir, "id_prop.csv")
    # from jarvis.db.figshare import data
    # df = pd.DataFrame(data(name))
    if not os.path.exists("X_dft_3d.csv"):
        from jarvis.db.figshare import data

        d = data("dft_3d")
        df = pd.DataFrame(d)
        df["structure"] = df["atoms"].progress_apply(
            lambda x: (
                (Atoms.from_dict(x)).get_primitive_atoms
            ).pymatgen_converter()
        )
        X, failed = StructureFeaturizer(df)
        X.to_csv("X_dft_3d.csv")
    df = pd.read_csv("X_dft_3d.csv")
    with open(id_prop_dat, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    dataset = []
    n_outputs = []
    multioutput = False
    lists_length_equal = True

    def get_desc(id=""):
        return (df[df["jid"] == id][important_features]).values[0]

    X = []
    y = []
    ids = []

    for i in tqdm(data):
        info = {}
        file_name = i[0]
        file_path = os.path.join(root_dir, file_name)
        if file_format == "poscar":
            atoms = Atoms.from_poscar(file_path)
        elif file_format == "cif":
            atoms = Atoms.from_cif(file_path)
        elif file_format == "xyz":
            # Note using 500 angstrom as box size
            atoms = Atoms.from_xyz(file_path, box_size=500)
        elif file_format == "pdb":
            # Note using 500 angstrom as box size
            # Recommended install pytraj
            # conda install -c ambermd pytraj
            atoms = Atoms.from_pdb(file_path, max_lat=500)
        else:
            raise NotImplementedError(
                "File format not implemented", file_format
            )

        info["atoms"] = atoms.to_dict()
        info["jid"] = file_name

        tmp = [float(j) for j in i[1:]]  # float(i[1])
        if len(tmp) == 1:
            tmp = tmp[0]
        else:
            multioutput = True
        info["target"] = tmp  # float(i[1])
        n_outputs.append(info["target"])
        dataset.append(info)
        X.append(get_desc(file_name))
        y.append(tmp)
        ids.append(file_name)
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    id_test = ids[-n_test:]

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[-(n_val + n_test) : -n_test]
    y_val = y[-(n_val + n_test) : -n_test]

    X_test = X[-n_test:]
    y_test = y[-n_test:]

    return X_train, y_train, X_val, y_val, X_test, y_test, id_test


lgbm = lgb.LGBMRegressor(
    # device="gpu",
    n_estimators=1740,
    learning_rate=0.040552334327414057,
    num_leaves=291,
    max_depth=16,
    min_data_in_leaf=14,
)
important_features = [
    "mean CN_VoronoiNN",
    "mean ordering parameter shell 1",
    "mean neighbor distance variation",
    "avg_dev CN_VoronoiNN",
    "mean local difference in NValence",
    "MagpieData mean NpUnfilled",
    "MagpieData mean NsUnfilled",
    "minimum local difference in Number",
    "MagpieData mode GSmagmom",
    "minimum local difference in Column",
    "MagpieData mode NfUnfilled",
    "MagpieData mode GSbandgap",
    "MagpieData maximum MeltingT",
    "avg_dev local difference in NdValence",
    "minimum local difference in NpUnfilled",
    "MagpieData maximum CovalentRadius",
    "MagpieData mode NValence",
    "MagpieData range NdUnfilled",
    "range local difference in NfValence",
    "avg_dev local difference in CovalentRadius",
    "minimum local difference in NdValence",
    "MagpieData mean NUnfilled",
    "MagpieData minimum AtomicWeight",
    "MagpieData mode NdUnfilled",
    "minimum local difference in NdUnfilled",
    "MagpieData mean MeltingT",
    "avg_dev local difference in NValence",
    "minimum local difference in MeltingT",
    "range local difference in NUnfilled",
    "MagpieData minimum NValence",
    "MagpieData minimum NsUnfilled",
    "minimum local difference in NpValence",
    "mean ordering parameter shell 3",
    "MagpieData minimum GSvolume_pa",
    "minimum local difference in GSvolume_pa",
    "MagpieData maximum Column",
    "frac d valence electrons",
    "MagpieData mode NpUnfilled",
    "avg_dev local difference in GSbandgap",
    "MagpieData minimum NdValence",
    "minimum local difference in CovalentRadius",
    "MagpieData avg_dev Row",
    "MagpieData minimum Electronegativity",
    "0-norm",
    "MagpieData maximum SpaceGroupNumber",
    "MagpieData range Electronegativity",
    "compound possible",
    "range local difference in Column",
    "MagpieData mode NsValence",
    "MagpieData mode NfValence",
    "minimum local difference in NsUnfilled",
    "MagpieData mode NUnfilled",
    "minimum neighbor distance variation",
    "MagpieData mean MendeleevNumber",
    "MagpieData avg_dev GSvolume_pa",
    "minimum local difference in GSmagmom",
    "minimum local difference in GSbandgap",
    "frac s valence electrons",
    "MagpieData minimum NfValence",
    "MagpieData maximum Row",
    "MagpieData minimum GSmagmom",
    "MagpieData range NpUnfilled",
    "range local difference in Row",
    "avg_dev local difference in NsValence",
    "MagpieData minimum GSbandgap",
    "mean local difference in SpaceGroupNumber",
    "MagpieData minimum NdUnfilled",
    "MagpieData minimum NUnfilled",
    "minimum local difference in NfUnfilled",
    "minimum local difference in NfValence",
    "MagpieData minimum NpUnfilled",
    "MagpieData mode NsUnfilled",
    "avg_dev local difference in MendeleevNumber",
    "max relative bond length",
    "avg_dev local difference in AtomicWeight",
    "10-norm",
    "avg_dev neighbor distance variation",
    "minimum local difference in NUnfilled",
    "MagpieData minimum NfUnfilled",
    "MagpieData mode Column",
    "MagpieData avg_dev MendeleevNumber",
    "MagpieData mode SpaceGroupNumber",
    "range local difference in NfUnfilled",
    "MagpieData mode GSvolume_pa",
    "min relative bond length",
    "MagpieData maximum NdValence",
    "maximum CN_VoronoiNN",
    "avg_dev local difference in NpValence",
    "MagpieData avg_dev GSmagmom",
    "avg_dev local difference in NpUnfilled",
]


lgbm = lgb.LGBMRegressor(
    # device="gpu",
    n_estimators=1170,
    learning_rate=0.15375236057119931,
    num_leaves=273,
)

prop = ["exfoliation_energy"]
# lgbm = lgb.LGBMRegressor()

for prop in props:
    fname = "AI-SinglePropertyPrediction-" + prop + "-dft_3d-test-mae.csv"
    json_zip = (
        "../../dataset/AI/SinglePropertyPrediction/dft_3d_"
        + prop
        + ".json.zip"
    )
    temp2 = "dft_3d" + "_" + prop + ".json"
    zp = zipfile.ZipFile(json_zip)
    train_val_test = json.loads(zp.read(temp2))
    output_path = "DataDir-" + prop
    train = train_val_test["train"]
    val = {}
    if "val" in train_val_test:
        val = train_val_test["val"]
    test = train_val_test["test"]
    cwd = os.getcwd()

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # os.chdir(output_path)
    id_prop = os.path.join(output_path, "id_prop.csv")
    info = {}
    dat = data("dft_3d")
    id_tag = "jid"
    for i in dat:
        info[i[id_tag]] = Atoms.from_dict(i["atoms"])
    f = open(id_prop, "w")
    for i, j in train.items():
        line = str(i) + "," + str(j) + "\n"
        f.write(line)
        pos_name = os.path.join(output_path, str(i))
        info[i].write_poscar(pos_name)
    for i, j in val.items():
        line = str(i) + "," + str(j) + "\n"
        f.write(line)
        pos_name = os.path.join(output_path, str(i))
        info[i].write_poscar(pos_name)

    for i, j in test.items():
        line = str(i) + "," + str(j) + "\n"
        f.write(line)
        pos_name = os.path.join(output_path, str(i))
        info[i].write_poscar(pos_name)
    f.close()
    n_train = len(train)
    n_val = len(val)
    n_test = len(test)

    print("number of training samples", len(train))
    print("number of validation samples", len(val))
    print("number of test samples", len(test))

    t1 = time.time()
    X_train, y_train, X_val, y_val, X_test, y_test, id_test = load_dataset(
        name="cfid_3d",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        root_dir=output_path,
        target=prop,
    )

    lgbm.fit(X_train, y_train)
    pred = lgbm.predict(X_test)
    reg_sc = regr_scores(y_test, pred)
    print(prop, reg_sc["mae"])

    f = open(fname, "w")
    line = "id,prediction\n"
    f.write(line)
    for j, k in zip(id_test, pred):
        line = str(j) + "," + str(k) + "\n"
        f.write(line)
    f.close()
    t2 = time.time()
    print("Time", t2 - t1)
    cmd = "zip " + fname + ".zip " + fname
    os.system(cmd)
