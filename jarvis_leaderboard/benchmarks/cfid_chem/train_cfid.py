from jarvis.ai.descriptors.cfid import CFID
import argparse
import csv
import os
import sys
from tqdm import tqdm
import lightgbm as lgb
import pandas as pd
from jarvis.core.atoms import Atoms
from sklearn.metrics import mean_absolute_error
import numpy as np
tqdm.pandas()



parser = argparse.ArgumentParser(
    description="CFID descriptors."
)
parser.add_argument(
    "--root_dir",
    default="./",
    help="Folder with id_props.csv, structure files",
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)


parser.add_argument(
    "--classification_threshold",
    default=None,
    help="Floating point threshold for converting into 0/1 class"
    + ", use only for classification tasks",
)


parser.add_argument(
    "--output_dir", default="./", help="Folder to save outputs",
)


parser.add_argument(
    "--n_train", default="650", help="number of training samples.",
)

parser.add_argument(
    "--n_val", default="81", help="number of validation samples.",
)

parser.add_argument(
    "--n_test", default="81", help="number of test samples.",
)

lgbm = lgb.LGBMRegressor(
    # device="gpu",
    n_estimators=1170,
    learning_rate=0.15375236057119931,
    num_leaves=273,
)

lgbm = lgb.LGBMRegressor()
if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    root_dir = args.root_dir
    n_train = int(args.n_train)
    n_val = int(args.n_val)
    n_test = int(args.n_test)

    id_prop_dat = os.path.join(root_dir, "id_prop.csv")
    file_format=args.file_format
    with open(id_prop_dat, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    dataset = []
    n_outputs = []
    multioutput = False
    lists_length_equal = True
    def get_desc(atoms=[]):
        return CFID(atoms).get_comp_descp(jrdf=False, jrdf_adf=False)
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
        X.append(get_desc(atoms))
        y.append(tmp)
        ids.append(file_name)
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[-(n_val + n_test) : -n_test]
    y_val = y[-(n_val + n_test) : -n_test]

    X_test = X[-n_test:]
    y_test = y[-n_test:]

    id_test = ids[-n_test:]
    #df = pd.DataFrame(dataset)
    #df['cfid']=df['atoms'].progress_apply(lambda x:CFID(Atoms.from_dict(x)).get_comp_descp(jrdf=False, jrdf_adf=False))
    #df_train = df[:n_train]
    #df_val = df[-(n_val + n_test) : -n_test]
    #df_test = df[-n_test:]
    #print (df)
    lgbm.fit(X_train,y_train)
    val_pred = lgbm.predict(X_val)
    test_pred = lgbm.predict(X_test)
    f=open('pred_test.csv','w')
    f.write('id,prediction\n')
    for i,j in zip(id_test,test_pred):
       line=str(i)+','+str(j)+'\n'
       f.write(line)
    f.close()
    print ('MAE Val',mean_absolute_error(y_val,val_pred))
    print ('MAE Test',mean_absolute_error(y_test,test_pred))
