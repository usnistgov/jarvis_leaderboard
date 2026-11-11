"""QV-JLB-Blend-v1 for a JARVIS-Leaderboard SinglePropertyPrediction task.

Composition-only. For each material the only input is its composition (the
formula, or the element list of the stored structure); no geometry is used.

Features: element fractions, plus matminer's Magpie statistics
(ElementProperty.from_preset("magpie", impute_nan=True)); Magpie columns with zero
variance over all rows of the task are dropped. No target is used.

Model: four fixed regressors averaged, with no hyperparameter search:
  RandomForestRegressor(n_estimators=500)            on Magpie
  LGBMRegressor(n_estimators=1200, lr=0.05, ...)      on fractions + Magpie
  CatBoostRegressor(iterations=1000, depth=6, ...)    on fractions + Magpie
  ExtraTreesRegressor(n_estimators=500)               on fractions + Magpie
Fit on the benchmark's train ids (plus val ids where the task has them), then the
test ids are predicted once.

    python run.py <task> [--bench-dir DIR] [--out-dir DIR]

<task> is a benchmark name such as mxene275_formation_energy or
halide_peroskites_HSE_gap. The benchmark file is read from --bench-dir
(default: this repository's jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction)
and the dataset through jarvis-tools (jarvis.db.figshare.data).
"""

import argparse
import ast
import csv
import io
import json
import os
import zipfile

import numpy as np

# task -> (dataset name in jarvis-tools, id field, target field, property tag in the file name)
TASKS = {
    "mxene275_formation_energy": ("mxene275", "id", "formation_energy", "formation_energy"),
    "halide_peroskites_HSE_gap": ("halide_peroskites", "id", "HSE_gap", "HSE_gap"),
}
THREADS = 4


def composition(row):
    from pymatgen.core import Composition
    for f in ("formula", "composition"):
        if row.get(f):
            return Composition(str(row[f]))
    atoms = row["atoms"]
    if isinstance(atoms, str):
        atoms = ast.literal_eval(atoms)
    counts = {}
    for e in atoms["elements"]:
        counts[str(e)] = counts.get(str(e), 0) + 1
    return Composition(counts)


def features(comps):
    from matminer.featurizers.composition import ElementProperty
    elements = sorted({str(e) for c in comps for e in c.elements})
    frac = np.array([[c.get_atomic_fraction(e) for e in elements] for c in comps], dtype=float)
    ep = ElementProperty.from_preset("magpie", impute_nan=True)
    ep.set_n_jobs(1)
    mag = np.array([ep.featurize(c) for c in comps], dtype=float)
    mag = mag[:, mag.std(axis=0) > 0]
    for m in (frac, mag):
        np.nan_to_num(m, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return mag, np.hstack([frac, mag])


def blend(mag, wide, y, tr, te):
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    members = [
        (RandomForestRegressor(n_estimators=500, n_jobs=THREADS, random_state=0), mag),
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

    ds_name, id_field, target_field, prop = TASKS[a.task]
    zf = zipfile.ZipFile(os.path.join(a.bench_dir, f"{a.task}.json.zip"))
    split = json.loads(zf.read(zf.namelist()[0]))
    rows = {str(r[id_field]): r for r in load_dataset(ds_name)}

    parts = ("train", "val") if "val" in split else ("train",)
    train_ids = [str(i) for p in parts for i in split[p]]
    test_ids = [str(i) for i in split["test"]]
    comps = [composition(rows[i]) for i in train_ids + test_ids]
    y = np.array([float(rows[i][target_field]) for i in train_ids + test_ids])
    mag, wide = features(comps)
    tr = np.arange(len(train_ids))
    te = np.arange(len(train_ids), len(comps))
    pred = blend(mag, wide, y, tr, te)

    name = f"AI-SinglePropertyPrediction-{prop}-{ds_name}-test-mae.csv"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "target", "prediction"])
    for i, t, p in zip(test_ids, y[te], pred):
        w.writerow([i, f"{t:.6f}", f"{p:.6f}"])
    os.makedirs(a.out_dir, exist_ok=True)
    with zipfile.ZipFile(os.path.join(a.out_dir, name + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, buf.getvalue())
    mae = float(np.mean(np.abs(pred - y[te])))
    print(f"{a.task}: {len(train_ids)} training ids, {len(test_ids)} test ids, test MAE {mae:.6f}")


if __name__ == "__main__":
    main()
