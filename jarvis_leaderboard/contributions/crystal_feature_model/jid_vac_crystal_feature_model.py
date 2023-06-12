import sys

import pandas as pd
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from jarvis.db.jsonutils import loadjson
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.composition import Composition

# Initialize CrystalNN
cnn = CrystalNN()

# get vacancy database
d = data('vacancydb')
df = pd.DataFrame(d)

# get dft_d3 database
dft_3d = data("dft_3d")
dft_3d_df = pd.DataFrame(dft_3d)

# get cohesive energy (Ec) dataframe
Ec_df = pd.read_csv("element_cohesive_energies.csv")
Ec_O = Ec_df.loc[Ec_df["Element"] == "O", "Per Atom"].values[0]

# get training and testing ids
train_test = loadjson("vacancydb_oxides_ef_train_test.json")
train_ids = list(train_test["train"].keys())
test_ids = list(train_test["test"].keys())

# get training and testing dataframes
train_df = df[df["id"].isin(train_ids)].copy()
test_df = df[df["id"].isin(test_ids)].copy()

# get crystal features
for i, row in train_df.iterrows():
    formula = row["bulk_formula"]

    # get non-oxygen element (M)
    M = [i.symbol for i in Composition(formula).elements if i.symbol != "O"]
    if len(M) == 1:
        M = M[0]
    else:
        sys.exit("More than one non-oxygen element in the formula")

    # get M oxidation state
    N_M = Composition(formula).get_el_amt_dict()[M]
    N_O = Composition(formula).get_el_amt_dict()["O"]
    n_M = 2 * N_O / N_M

    # get formation energy, band gap, and energy above the convex hull
    dEf = dft_3d_df.loc[dft_3d_df["jid"] == row["jid"], "formation_energy_peratom"].values[0] * (N_M + N_O)
    Eg_optb88vdw = dft_3d_df.loc[dft_3d_df["jid"] == row["jid"], "optb88vdw_bandgap"].values[0]
    # Eg_mbj = dft_3d_df.loc[dft_3d_df["jid"] == row["jid"], "mbj_bandgap"].values[0]
    # Eg_hse = dft_3d_df.loc[dft_3d_df["jid"] == row["jid"], "hse_gap"].values[0]
    Ehull = dft_3d_df.loc[dft_3d_df["jid"] == row["jid"], "ehull"].values[0]

    # get cohesive energies
    Ec_M = Ec_df.loc[Ec_df["Element"] == M, "Per Atom"].values[0]
    Ec = -dEf + N_M * Ec_M + N_O * Ec_O

    # get number of O-M bonds
    structure = Atoms.from_dict(row["bulk_atoms"]).pymatgen_converter()
    structure.add_oxidation_state_by_element({M: n_M, "O": -2})
    i_O = [i for i in range(len(structure)) if structure[i].specie.symbol == "O"][0]  # assumes only one oxygen vacancy
    cation = f"{M}{int(n_M)}+"
    CN_O = cnn.get_cn_dict(structure, i_O)[cation]

    # get crystal bond dissociation energy sum (Eb)
    Eb = Ec / 2  # cohesive energy per O2-

    # get crystal reduction potential
    m_M = n_M - 1
    if m_M % 2 == 0:
        N_M_red = 1
    else:
        N_M_red = 2
    N_O_red = m_M * N_M_red / 2
    formula_red = f"{M}{int(N_M_red)}O{int(N_O_red)}"
    dEf_red = dft_3d_df.loc[dft_3d_df["formula"] == formula_red, "formation_energy_peratom"].min() * (N_M_red + N_O_red)
    Er = dEf_red / N_M_red - dEf / N_M
    Vr = -Er / (n_M - m_M)
    break
