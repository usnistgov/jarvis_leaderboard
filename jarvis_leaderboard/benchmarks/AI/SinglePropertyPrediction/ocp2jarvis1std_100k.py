# conda activate ocp-models
from jarvis.core.atoms import Atoms
from jarvis.core.specie import atomic_numbers_to_symbols
from ocpmodels.datasets import SinglePointLmdbDataset
from jarvis.db.jsonutils import dumpjson
from collections import defaultdict

# dataset = SinglePointLmdbDataset({"src": "/wrk/knc6/oc/ocp/data/is2re/10k/train/data.lmdb"})
train = SinglePointLmdbDataset(
    {"src": "/wrk/knc6/oc/ocp/data/is2re/100k/train/data.lmdb"}
)
val = SinglePointLmdbDataset(
    {"src": "/wrk/knc6/oc/ocp/data/is2re/all/val_id/data.lmdb"}
)
test = SinglePointLmdbDataset(
    {"src": "/wrk/knc6/oc/ocp/data/is2re/all/test_id/data.lmdb"}
)
print("n_train=", len(train))
print("n_val=", len(val))
print("n_test=", len(test))
my_dir = "/wrk/knc6/oc/oc2/ocp/data/ALIGNN/10k/DataDir/"
my_dir = "/wrk/knc6/Software/alignn_calc/jarvis_leaderboard/jarvis_leaderboard/contributions/alignn_model/OCP/100k/DataDir/"

mem = []
xtrain = defaultdict()
xval = defaultdict()
xtest = defaultdict()

count = 0
f = open(my_dir + "id_prop.csv", "w")
# target_mean= -1.525913953781128
# target_std= 2.279365062713623
for i in train:
    lattice_mat = i.cell.numpy()[0].tolist()
    atomic_numbers = i.atomic_numbers.numpy().tolist()
    elements = atomic_numbers_to_symbols(atomic_numbers)
    pos = i.pos.numpy().tolist()
    atoms = Atoms(
        lattice_mat=lattice_mat, elements=elements, coords=pos, cartesian=True
    )
    # relaxed_energy = (i.y_relaxed-target_mean)/target_std
    relaxed_energy = i.y_relaxed
    count += 1
    fname = str("ocp-") + str(count)  # + ".vasp"
    pth = my_dir + fname
    line = fname + "," + str(relaxed_energy) + "\n"
    f.write(line)
    atoms.write_poscar(pth)
    info = {}
    info["id"] = fname
    info["atoms"] = atoms.to_dict()
    info["relaxed_energy"] = relaxed_energy
    mem.append(info)
    xtrain[fname] = relaxed_energy

for ii, i in enumerate(val):
    # if ii<1251:
    lattice_mat = i.cell.numpy()[0].tolist()
    atomic_numbers = i.atomic_numbers.numpy().tolist()
    elements = atomic_numbers_to_symbols(atomic_numbers)
    pos = i.pos.numpy().tolist()
    atoms = Atoms(
        lattice_mat=lattice_mat, elements=elements, coords=pos, cartesian=True
    )
    relaxed_energy = i.y_relaxed
    # relaxed_energy = (i.y_relaxed-target_mean)/target_std
    count += 1
    fname = str("ocp-") + str(count)  # + ".vasp"
    pth = my_dir + fname
    line = fname + "," + str(relaxed_energy) + "\n"
    f.write(line)
    atoms.write_poscar(pth)
    info = {}
    info["id"] = fname
    info["atoms"] = atoms.to_dict()
    info["relaxed_energy"] = relaxed_energy
    mem.append(info)
    xval[fname] = relaxed_energy


for ii, i in enumerate(val):
    # if ii<1251:
    # for i in test:
    lattice_mat = i.cell.numpy()[0].tolist()
    atomic_numbers = i.atomic_numbers.numpy().tolist()
    elements = atomic_numbers_to_symbols(atomic_numbers)
    pos = i.pos.numpy().tolist()
    atoms = Atoms(
        lattice_mat=lattice_mat, elements=elements, coords=pos, cartesian=True
    )
    # relaxed_energy = (i.y_relaxed-target_mean)/target_std
    relaxed_energy = i.y_relaxed
    count += 1
    fname = str("ocp-") + str(count)  # + ".vasp"
    pth = my_dir + fname
    line = fname + "," + str(relaxed_energy) + "\n"
    f.write(line)
    atoms.write_poscar(pth)
    info = {}
    info["id"] = fname
    info["atoms"] = atoms.to_dict()
    info["relaxed_energy"] = relaxed_energy
    mem.append(info)
    xtest[fname] = relaxed_energy

m = {}
m["train"] = xtrain
m["val"] = xval
m["test"] = xtest
dumpjson(data=mem, filename="ocp100k.json")
dumpjson(data=m, filename="ocp100k_relaxed_energy.json")
f.close()
