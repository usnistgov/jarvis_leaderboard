"""QV-JLB-Site-Blend-v1 for the JARVIS-Leaderboard benchmark tinnet_O_ead.

The input is the stored structure of each slab with its adsorbed oxygen (the
jarvis-tools dataset tinnet_O). Features, computed with no target:

  * matminer's Magpie statistics (ElementProperty.from_preset("magpie",
    impute_nan=True)) and element fractions of three compositions per structure:
      - the whole slab;
      - shell 1: the metal atoms within 1.25 x the closest metal distance of the
        oxygen (the binding site), periodic images included;
      - shell 2: the metal atoms within 2.0 x that distance.
    Magpie columns with zero variance over all rows are dropped.
  * eight numbers for the site's geometry: the sizes of shells 1 and 2, the closest
    metal distance, the mean and largest shell-1 distance, the mean shell-2 distance,
    the number of elements in shell 1, and the oxygen's height above the highest
    metal atom along the surface normal (a x b).

Model: four fixed regressors averaged, with no hyperparameter search:
  RandomForestRegressor(n_estimators=500)           on Magpie (three sets) + geometry
  LGBMRegressor(n_estimators=1200, lr=0.05, ...)     on fractions + the above
  CatBoostRegressor(iterations=1000, depth=6, ...)   on fractions + the above
  ExtraTreesRegressor(n_estimators=500)              on fractions + the above
Fit on the benchmark's train and val ids, then the test ids are predicted once.

    python run.py tinnet_O_ead [--bench-dir DIR] [--out-dir DIR]

The benchmark file is read from --bench-dir (default: this repository's
jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction) and the dataset through
jarvis-tools (jarvis.db.figshare.data).
"""

import argparse
import ast
import csv
import io
import json
import os
import zipfile
from collections import Counter

import numpy as np

# task -> (dataset name in jarvis-tools, id field, target field, property tag, binding element)
TASKS = {"tinnet_O_ead": ("tinnet_O", "id", "ead", "ead", "O")}
NONMETAL = {"H", "C", "N", "O"}
RADIUS, SHELL1, SHELL2 = 6.0, 1.25, 2.0
THREADS = 4
ZIP_TIME = (2026, 11, 12, 0, 0, 0)     # fixed, so the same predictions give the same zip bytes


def atoms_of(row):
    atoms = row["atoms"]
    return ast.literal_eval(atoms) if isinstance(atoms, str) else atoms


def composition(row):
    from pymatgen.core import Composition
    counts = {}
    for e in atoms_of(row)["elements"]:
        counts[str(e)] = counts.get(str(e), 0) + 1
    return Composition(counts)


def site(row, bind):
    """(shell-1 composition, shell-2 composition, eight geometry numbers)."""
    from pymatgen.core import Composition, Lattice, Structure
    a = atoms_of(row)
    s = Structure(Lattice(np.asarray(a["lattice_mat"], dtype=float)), list(a["elements"]),
                  np.asarray(a["coords"], dtype=float),
                  coords_are_cartesian=bool(a.get("cartesian", False)))
    best = None
    for i, x in enumerate(s):
        if x.specie.symbol != bind:
            continue
        nbrs = [n for n in s.get_neighbors(x, RADIUS) if n.specie.symbol not in NONMETAL]
        if nbrs:
            dmin = min(n.nn_distance for n in nbrs)
            if best is None or dmin < best[0]:
                best = (dmin, i, nbrs)
    dmin, i, nbrs = best
    s1 = [n for n in nbrs if n.nn_distance <= SHELL1 * dmin]
    s2 = [n for n in nbrs if n.nn_distance <= SHELL2 * dmin]
    c1 = Counter(n.specie.symbol for n in s1)
    c2 = Counter(n.specie.symbol for n in s2)
    m = s.lattice.matrix
    normal = np.cross(m[0], m[1])
    normal = normal / np.linalg.norm(normal)
    top = max(float(np.dot(x.coords, normal)) for x in s if x.specie.symbol not in NONMETAL)
    d1 = [n.nn_distance for n in s1]
    d2 = [n.nn_distance for n in s2]
    geom = [len(s1), len(s2), dmin, float(np.mean(d1)), float(np.max(d1)), float(np.mean(d2)),
            len(c1), float(np.dot(s[i].coords, normal)) - top]
    return Composition(dict(c1)), Composition(dict(c2)), geom


def magpie_and_fractions(comps):
    from matminer.featurizers.composition import ElementProperty
    elements = sorted({str(e) for c in comps for e in c.elements})
    frac = np.array([[c.get_atomic_fraction(e) for e in elements] for c in comps], dtype=float)
    ep = ElementProperty.from_preset("magpie", impute_nan=True)
    ep.set_n_jobs(1)
    mag = np.array([ep.featurize(c) for c in comps], dtype=float)
    mag = mag[:, mag.std(axis=0) > 0]
    for x in (frac, mag):
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return mag, frac


def features(rows, bind):
    slab = [composition(r) for r in rows]
    sites = [site(r, bind) for r in rows]
    m0, f0 = magpie_and_fractions(slab)
    m1, f1 = magpie_and_fractions([x[0] for x in sites])
    m2, f2 = magpie_and_fractions([x[1] for x in sites])
    geom = np.asarray([x[2] for x in sites], dtype=float)
    plain = np.hstack([m0, m1, m2, geom])
    return plain, np.hstack([f0, f1, f2, plain])


def blend(plain, wide, y, tr, te):
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    members = [
        (RandomForestRegressor(n_estimators=500, n_jobs=THREADS, random_state=0), plain),
        (lgb.LGBMRegressor(objective="l2", n_estimators=1200, learning_rate=0.05, num_leaves=31,
                           min_child_samples=10, colsample_bytree=0.5, subsample=0.8,
                           subsample_freq=1, reg_lambda=1.0, n_jobs=THREADS, verbose=-1,
                           random_state=1), wide),
        (CatBoostRegressor(loss_function="RMSE", iterations=1000, learning_rate=0.05, depth=6,
                           l2_leaf_reg=3.0, rsm=0.5, thread_count=THREADS, verbose=0,
                           allow_writing_files=False, random_seed=2), wide),
        (ExtraTreesRegressor(n_estimators=500, n_jobs=THREADS, random_state=3), wide),
    ]
    preds = []
    for est, X in members:
        est.fit(X[tr], y[tr])
        preds.append(np.asarray(est.predict(X[te]), dtype=float).ravel())
    return np.mean(np.vstack(preds), axis=0)


def load_dataset(name):
    zip_dir = os.environ.get("QV_DATASET_ZIP_DIR")     # a local copy, for an offline re-run
    if zip_dir:
        zf = zipfile.ZipFile(os.path.join(zip_dir, f"ds_{name}", f"{name}.zip"))
        return json.loads(zf.read(zf.namelist()[0]))
    from jarvis.db.figshare import data
    return data(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TASKS))
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--bench-dir", default=os.path.join(
        here, "..", "..", "benchmarks", "AI", "SinglePropertyPrediction"))
    ap.add_argument("--out-dir", default=here)
    a = ap.parse_args()

    ds_name, id_field, target_field, prop, bind = TASKS[a.task]
    zf = zipfile.ZipFile(os.path.join(a.bench_dir, f"{a.task}.json.zip"))
    split = json.loads(zf.read(zf.namelist()[0]))
    rows = {str(r[id_field]): r for r in load_dataset(ds_name)}

    parts = ("train", "val") if "val" in split else ("train",)
    train_ids = [str(i) for p in parts for i in split[p]]
    test_ids = [str(i) for i in split["test"]]
    use = [rows[i] for i in train_ids + test_ids]
    y = np.array([float(r[target_field]) for r in use])
    plain, wide = features(use, bind)
    tr = np.arange(len(train_ids))
    te = np.arange(len(train_ids), len(use))
    pred = blend(plain, wide, y, tr, te)

    name = f"AI-SinglePropertyPrediction-{prop}-{ds_name}-test-mae.csv"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["id", "target", "prediction"])
    for i, t, p in zip(test_ids, y[te], pred):
        w.writerow([i, f"{t:.6f}", f"{p:.6f}"])
    os.makedirs(a.out_dir, exist_ok=True)
    info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(os.path.join(a.out_dir, name + ".zip"), "w") as z:
        z.writestr(info, buf.getvalue())
    mae = float(np.mean(np.abs(pred - y[te])))
    print(f"{a.task}: {len(train_ids)} training ids, {len(test_ids)} test ids, test MAE {mae:.6f}")


if __name__ == "__main__":
    main()
