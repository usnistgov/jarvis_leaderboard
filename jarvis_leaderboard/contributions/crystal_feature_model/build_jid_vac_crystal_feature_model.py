import math

import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from jarvis.db.jsonutils import loadjson
from matplotlib import pyplot as plt
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.structure_analyzer import OxideType
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Species
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error


def get_formulas(species):
    formulas = []
    for specie in species:
        if specie.oxi_state % 2 == 0:
            nu_O = int(specie.oxi_state / 2)
            if nu_O == 1:
                formula = f"{specie.symbol}O"
            else:
                formula = f"{specie.symbol}O{nu_O}"
        else:
            nu_O = int(specie.oxi_state)
            if nu_O == 1:
                formula = f"{specie.symbol}2O"
            else:
                formula = f"{specie.symbol}2O{nu_O}"
        formulas.append(formula)
    return formulas


def get_formation_energy(formula, jid=None):
    num_atoms = Composition(formula).num_atoms
    if jid is None:
        dEf = dft_3d_df.loc[dft_3d_df["formula"] == formula, "formation_energy_peratom"].min() * num_atoms
    else:
        try:
            dEf = dft_3d_df.loc[dft_3d_df["jid"] == jid, "formation_energy_peratom"].values[0] * num_atoms
        except IndexError:
            dEf = dft_2d_df.loc[dft_2d_df["jid"] == jid, "formation_energy_peratom"].values[0] * num_atoms
    return dEf


def get_cohesive_energy(formula, M, jid=None):
    dEf = get_formation_energy(formula, jid=jid)
    N_M = Composition(formula).get_el_amt_dict()[M.symbol]
    N_O = Composition(formula).get_el_amt_dict()["O"]
    Ec_M = Ec_df.loc[Ec_df["Element"] == M.symbol, "Per Atom"].values[0]
    Ec = -dEf + N_M * Ec_M + N_O * Ec_O
    return Ec


def get_coordination_number(formula, jid=None):
    dEf = get_formation_energy(formula, jid=jid)
    num_atoms = Composition(formula).num_atoms
    dEf_per_atom = round(dEf / num_atoms, 5)  # avoids floating point errors
    if jid is None:
        condition = (dft_3d_df["formula"] == formula) & (dft_3d_df["formation_energy_peratom"] == dEf_per_atom)
    else:
        condition = (dft_3d_df["jid"] == jid) & (dft_3d_df["formation_energy_peratom"] == dEf_per_atom)
    try:
        atoms_formula = Atoms.from_dict(dft_3d_df.loc[condition, "atoms"].values[0])
    except IndexError:
        condition = (dft_2d_df["jid"] == jid) & (dft_2d_df["formation_energy_peratom"] == dEf_per_atom)
        atoms_formula = Atoms.from_dict(dft_2d_df.loc[condition, "atoms"].values[0])
    try:
        structure_formula = BVAnalyzer().get_oxi_state_decorated_structure(atoms_formula.pymatgen_converter())
    except ValueError:
        structure_formula = atoms_formula.pymatgen_converter()
        if structure_formula.composition.reduced_formula == "AlO2":
            oxi_states = []
            for j, site in enumerate(structure_formula.sites):
                if site.specie.symbol == "Al":
                    oxi_states.append(3)
                if site.specie.symbol == "O":
                    if j % 2 == 0:
                        oxi_states.append(-2)
                    else:
                        oxi_states.append(-1)
        structure_formula.add_oxidation_state_by_site(oxi_states)
    i_O = [i for i in range(len(structure_formula)) if structure_formula[i].specie.symbol == "O"][0]
    CN = CrystalNN().get_cn_dict(structure_formula, i_O)
    return CN


def get_crystal_bond_dissociation_energy(formula, CN, M, Ec):
    N_O = Composition(formula).get_el_amt_dict()["O"]
    N_b = N_O * CN[M.to_pretty_string()]
    Eb = Ec / N_b
    return Eb


def get_next_highest_oxidation_state(M):
    n = M.oxi_state
    try:
        m = int(Vr_df.loc[(Vr_df["elem"] == M.symbol) & (Vr_df["n"] == M.oxi_state), "m"].values[0])
    except IndexError:
        ms = np.arange(n - 1, -1, -1)
        for m in ms:
            if m != 0:
                formula_m = get_formulas([Species(M.symbol, m)])[0]
            else:
                formula_m = M.symbol
            if not math.isnan(get_formation_energy(formula_m)):
                break
    return m


def get_crystal_reduction_potential(formula_n, M, jid=None):
    # get formula of reduced metal oxide
    n = M.oxi_state
    m = get_next_highest_oxidation_state(M)
    if m != 0:
        formula_m = get_formulas([Species(M.symbol, m)])[0]
    else:
        formula_m = M.symbol

    # if M = Co4+, m = (8/3)+ and formula_m = Co3O4
    if M.symbol == "Co" and n == 4:
        m = 8 / 3
        formula_m = "Co3O4"

    # get formation energies
    dEf_n = get_formation_energy(formula_n, jid=jid)
    dEf_m = get_formation_energy(formula_m)

    # get number of non-O atoms in formula
    N_M_n = Composition(formula_n).get_el_amt_dict()[M.symbol]
    N_M_m = Composition(formula_m).get_el_amt_dict()[M.symbol]

    # get crystal reduction energy
    Er = dEf_m / N_M_m - dEf_n / N_M_n

    # get crystal reduction potential
    Vr = -Er / (n - m)
    return Vr


def get_crystal_features(dataframe):
    for i, row in dataframe.iterrows():
        # get oxidation state decorated structure
        atoms = Atoms.from_dict(row["bulk_atoms"])
        structure = BVAnalyzer().get_oxi_state_decorated_structure(atoms.pymatgen_converter())

        # get non-O species (Ms)
        composition = structure.composition.reduced_composition
        Ms = [species for species in composition if species.symbol != "O"]

        # get formulas
        formulas = get_formulas(Ms)

        # get cohesive energies (Ecs)
        Ecs = [get_cohesive_energy(formula, M, jid=row["jid"]) for formula, M in zip(formulas, Ms)]

        # get O coordination numbers (CNs) assuming only one unique O atom
        CNs = [get_coordination_number(formula, jid=row["jid"]) for formula in formulas]

        # get crystal bond dissociation energies (Ebs)
        Ebs = [get_crystal_bond_dissociation_energy(formula, CN, M, Ec) for formula, CN, M, Ec in
               zip(formulas, CNs, Ms, Ecs)]

        # get crystal bond dissociation energy sum (sumEb)
        i_O = [j for j in range(len(structure)) if structure[j].specie.symbol == "O"][0]
        CN_structure = CrystalNN().get_cn_dict(structure, i_O)
        sumEb = 0
        for M, Eb in zip(Ms, Ebs):
            sumEb += CN_structure[M.to_pretty_string()] * Eb

        # get crystal reduction potentials (Vrs)
        Vrs = [get_crystal_reduction_potential(formula, M, jid=row["jid"]) for formula, M in zip(formulas, Ms)]
        maxVr = max(Vrs)

        # get band gap
        try:
            Eg = dft_3d_df.loc[dft_3d_df["jid"] == row["jid"], "optb88vdw_bandgap"].values[0]
        except IndexError:
            Eg = dft_2d_df.loc[dft_2d_df["jid"] == row["jid"], "optb88vdw_bandgap"].values[0]

        # get energy above the convex hull
        try:
            Ehull = dft_3d_df.loc[dft_3d_df["jid"] == row["jid"], "ehull"].values[0]
        except IndexError:
            Ehull = dft_2d_df.loc[dft_2d_df["jid"] == row["jid"], "ehull"].values[0]

        # add crystal features to dataframe
        dataframe.loc[i, "sumEb"] = sumEb
        dataframe.loc[i, "maxVr"] = maxVr
        dataframe.loc[i, "Eg"] = Eg
        dataframe.loc[i, "Ehull"] = Ehull
    return dataframe


# set random number generator
rng = np.random.default_rng(742023)

# get vacancy database
d = data('vacancydb')
df = pd.DataFrame(d)

# get dft_d3 database
dft_3d = data("dft_3d")
dft_3d_df = pd.DataFrame(dft_3d)

# get dft_2d database
dft_2d = data("dft_2d")
dft_2d_df = pd.DataFrame(dft_2d)

# get cohesive energy (Ec) dataframe
Ec_df = pd.read_csv("element_cohesive_energies.csv")
Ec_O = Ec_df.loc[Ec_df["Element"] == "O", "Per Atom"].values[0]

# get crystal bond dissociation energy (Eb) dataframe
Eb_df = pd.read_csv("Eb.csv")

# get crystal reduction potential (Vr) dataframe
Vr_df = pd.read_csv("Vr.csv")

# get training and testing ids
train_test = loadjson("vacancydb_oxides_ef_train_test.json")
train_ids = list(train_test["train"].keys())
test_ids = list(train_test["test"].keys())

# get training and testing dataframes
train_df = df[df["id"].isin(train_ids)].copy()
test_df = df[df["id"].isin(test_ids)].copy()

# remove AlO2 from training data
train_df = train_df[train_df["bulk_formula"] != "AlO2"].copy()

# remove peroxides from training data
for i, row in train_df.iterrows():
    train_df.loc[i, "oxide_type"] = OxideType(Atoms.from_dict(row["bulk_atoms"]).pymatgen_converter()).parse_oxide()[0]
train_df = train_df[train_df["oxide_type"] != "peroxide"].copy()

# get oxide types for testing data
for i, row in test_df.iterrows():
    test_df.loc[i, "oxide_type"] = OxideType(Atoms.from_dict(row["bulk_atoms"]).pymatgen_converter()).parse_oxide()[0]

# get crystal features
train_df = get_crystal_features(train_df)
test_df = get_crystal_features(test_df)

# train model
coefs = []
intercepts = []
valid_maes = []
for i in range(1000):
    indices = rng.choice(train_df.index, size=10, replace=False)
    train_X = train_df.loc[indices, ["sumEb", "maxVr", "Eg", "Ehull"]].values
    train_y = train_df.loc[indices, "ef"].values
    valid_X = train_df.loc[~train_df.index.isin(indices), ["sumEb", "maxVr", "Eg", "Ehull"]].values
    valid_y = train_df.loc[~train_df.index.isin(indices), "ef"].values
    cfm = linear_model.HuberRegressor(max_iter=1000).fit(train_X, train_y)
    coefs.append(cfm.coef_)
    intercepts.append(cfm.intercept_)
    valid_maes.append(mean_absolute_error(valid_y, cfm.predict(valid_X)))

# coefficient statistics
mean_coef = np.mean(coefs, axis=0)
std_coef = np.std(coefs, axis=0)

# intercept statistics
mean_intercept = np.mean(intercepts)
std_intercept = np.std(intercepts)

# write model to file
with open("crystal_feature_model.txt", "w") as f:
    for coef in mean_coef:
        f.write(f"{coef}\n")
    f.write(f"{mean_intercept}\n")

# validation statistics
mean_valid_mae = np.mean(valid_maes)
std_valid_mae = np.std(valid_maes)

# predict training y values
train_X = train_df[["sumEb", "maxVr", "Eg", "Ehull"]].values
train_y = train_df["ef"].values
train_y_pred = np.dot(train_X, mean_coef) + mean_intercept
train_mae = mean_absolute_error(train_y, train_y_pred)

# test model
test_X = test_df[["sumEb", "maxVr", "Eg", "Ehull"]].values
test_y = test_df["ef"].values
test_y_pred = np.dot(test_X, mean_coef) + np.mean(intercepts)
test_mae = mean_absolute_error(test_y, test_y_pred)

# parity plot
fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.plot(train_y_pred, train_y, "o", color="black", label="Training Set", alpha=0.5)
ax.plot(test_y_pred, test_y, "o", color="red", label="Testing Set", alpha=0.5)
ax.plot(np.arange(8), np.arange(8), "--", color="black")
ax.set_xlabel(
    f"{mean_coef[0]:.1f}$\Sigma E_b${mean_coef[1]:+.1f}$V_r${mean_coef[2]:+.1f}$E_g${mean_coef[3]:+.1f}$E_{{hull}}${np.mean(intercepts):+.1f} (eV)")
ax.set_ylabel("DFT $E_v$ (eV)")
ax.text(0.05, 0.9, f"MAE = {train_mae:.2f} eV", transform=ax.transAxes)
ax.text(0.05, 0.8, f"MAE = {test_mae:.2f} eV", transform=ax.transAxes, color="red")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("parity_plot.png")
