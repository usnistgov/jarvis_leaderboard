import sys

import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from jarvis.db.jsonutils import loadjson
from matplotlib import pyplot as plt
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.composition import Composition
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error


def get_formulas(species):
    formulas = []
    for specie in species:
        if specie.oxi_state % 2 == 0:
            formula = f"{specie.symbol}O{int(specie.oxi_state / 2)}"
        else:
            formula = f"{specie.symbol}2O{int(specie.oxi_state)}"
        formulas.append(formula)
    return formulas


def get_formation_energy(formula):
    num_atoms = Composition(formula).num_atoms
    dEf = dft_3d_df.loc[dft_3d_df["formula"] == formula, "formation_energy_peratom"].min() * num_atoms
    return dEf


def get_cohesive_energy(formula, M):
    dEf = get_formation_energy(formula)
    N_M = Composition(formula).get_el_amt_dict()[M.symbol]
    N_O = Composition(formula).get_el_amt_dict()["O"]
    Ec_M = Ec_df.loc[Ec_df["Element"] == M.symbol, "Per Atom"].values[0]
    Ec = -dEf + N_M * Ec_M + N_O * Ec_O
    return Ec


def get_coordination_number(formula):
    dEf = get_formation_energy(formula)
    num_atoms = Composition(formula).num_atoms
    condition = (dft_3d_df["formula"] == formula) & (dft_3d_df["formation_energy_peratom"] == dEf / num_atoms)
    atoms_formula = Atoms.from_dict(dft_3d_df.loc[condition, "atoms"].values[0])
    structure_formula = BVAnalyzer().get_oxi_state_decorated_structure(atoms_formula.pymatgen_converter())
    i_O = [i for i in range(len(structure_formula)) if structure_formula[i].specie.symbol == "O"][0]
    CN = CrystalNN().get_cn_dict(structure_formula, i_O)
    return CN


def get_crystal_bond_dissociation_energy(formula, CN, M, Ec):
    N_O = Composition(formula).get_el_amt_dict()["O"]
    N_b = N_O * CN[M.to_pretty_string()]
    Eb = Ec / N_b
    return Eb


def get_crystal_reduction_potential(formula, M):
    dEf = get_formation_energy(formula)
    N_M = Composition(formula).get_el_amt_dict()[M.symbol]
    Er = -dEf / N_M
    Vr = -Er / M.oxi_state
    return Vr


def get_crystal_features(dataframe):
    for i, row in dataframe.iterrows():
        # get oxidation state decorated structure
        atoms = Atoms.from_dict(row["bulk_atoms"])
        try:
            structure = BVAnalyzer().get_oxi_state_decorated_structure(atoms.pymatgen_converter())
        except ValueError:
            print(atoms)
            sys.exit()

        # get non-O species (Ms)
        composition = structure.composition.reduced_composition
        Ms = [species for species in composition if species.symbol != "O"]

        # get formulas
        formulas = get_formulas(Ms)

        # get formation energies (dEfs)
        dEfs = [get_formation_energy(formula) for formula in formulas]

        # get cohesive energies (Ecs)
        Ecs = [get_cohesive_energy(formula, M) for formula, M in zip(formulas, Ms)]

        # get O coordination numbers (CNs) assuming only one unique O atom
        CNs = [get_coordination_number(formula) for formula in formulas]

        # get crystal bond dissociation energies (Ebs)
        Ebs = [get_crystal_bond_dissociation_energy(formula, CN, M, Ec) for formula, CN, M, Ec in
               zip(formulas, CNs, Ms, Ecs)]

        # get crystal bond dissociation energy sum (sumEb)
        i_O = [i for i in range(len(structure)) if structure[i].specie.symbol == "O"][0]
        CN_structure = CrystalNN().get_cn_dict(structure, i_O)
        sumEb = 0
        for M, Eb in zip(Ms, Ebs):
            sumEb += CN_structure[M.to_pretty_string()] * Eb

        # get crystal reduction potentials (Vrs)
        Vrs = [get_crystal_reduction_potential(formula, M) for formula, M in zip(formulas, Ms)]
        maxVr = max(Vrs)

        # get band gap
        Eg = dft_3d_df.loc[dft_3d_df["jid"] == row["jid"], "optb88vdw_bandgap"].values[0]

        # get energy above the convex hull
        Ehull = dft_3d_df.loc[dft_3d_df["jid"] == row["jid"], "ehull"].values[0]

        # add crystal features to dataframe
        dataframe.loc[i, "sumEb"] = sumEb
        dataframe.loc[i, "maxVr"] = maxVr
        dataframe.loc[i, "Eg"] = Eg
        dataframe.loc[i, "Ehull"] = Ehull
    return dataframe


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
train_df = get_crystal_features(train_df)
print(train_df)
sys.exit()
test_df = get_crystal_features(test_df)

# train model
train_X = train_df[["Eb", "Vr", "Eg", "Ehull"]].values
train_y = train_df["ef"].values
huber = HuberRegressor().fit(train_X, train_y)
train_mae = mean_absolute_error(train_y, huber.predict(train_X))

# test model
test_X = test_df[["Eb", "Vr", "Eg", "Ehull"]].values
test_y = test_df["ef"].values
test_mae = mean_absolute_error(test_y, huber.predict(test_X))
print(test_mae)

# parity plot
fig, ax = plt.subplots()
ax.plot(train_y, huber.predict(train_X), "o", color="black")
for i, (yi, yi_hat) in enumerate(zip(train_y, huber.predict(train_X))):
    ax.annotate(f'{train_df.bulk_formula.values[i]}', (yi, yi_hat), textcoords="offset points", xytext=(0, 10),
                ha='center')
ax.plot(np.arange(8), np.arange(8), "--", color="black")
ax.set_xlabel("DFT $E_v$ (eV)")
ax.set_ylabel("CFM $E_v$ (eV)")
ax.set_title("Training Set")
ax.text(0.05, 0.95, f"MAE = {train_mae:.3f} eV", transform=ax.transAxes)
plt.savefig("parity_plot.png")
