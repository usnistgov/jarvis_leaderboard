"""Reproduce the fixed 3-seed XGBoost prediction from JARVIS structures.

Usage: python run.py --raw JDFT.json --split dft_3d_magmom_oszicar.json.zip
The output is the benchmark-named CSV and ZIP in this directory by default.
Test map values, raw material DFT properties, and public reference predictions
are not used to fit, choose, or change this model or its threshold.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import zipfile

import ijson
import numpy as np
import xgboost as xgb

import feature_enrich

HERE = Path(__file__).resolve().parent
NAME = "AI-SinglePropertyClass-magmom_oszicar-dft_3d-test-acc.csv"


def test_ids(path):
    ids = []
    with zipfile.ZipFile(path) as archive, archive.open(archive.namelist()[0]) as stream:
        for prefix, event, value in ijson.parse(stream):
            if prefix == "test" and event == "map_key":
                ids.append(str(value))
            elif prefix == "test" and event == "end_map":
                break
    if len(ids) != 5222 or len(ids) != len(set(ids)):
        raise ValueError("Expected exactly 5222 unique official test IDs")
    return ids


def row(item):
    jid, atoms = item
    return jid, feature_enrich.featurize_atoms(atoms, jid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", default=HERE, type=Path)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    config = json.loads((HERE / "model_config.json").read_text(encoding="utf-8"))
    if config["threshold"] != 0.475 or len(config["models"]) != 3:
        raise ValueError("Unexpected model configuration")
    ids = test_ids(args.split)
    atoms = feature_enrich.old_features.read_atoms(args.raw, set(ids))
    features = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for jid, vector in pool.map(row, ((jid, atoms[jid]) for jid in ids), chunksize=32):
            features[jid] = vector
    matrix = np.stack([features[jid] for jid in ids]).astype(np.float32)
    if matrix.shape != (5222, 772) or not np.isfinite(matrix).all():
        raise ValueError("Unexpected feature shape or non-finite value")
    probabilities = []
    for entry in config["models"]:
        path = HERE / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Model checksum mismatch")
        model = xgb.XGBClassifier()
        model.load_model(path)
        model.set_params(device="cpu", n_jobs=8)
        probabilities.append(model.predict_proba(matrix)[:, 1].astype(np.float64))
    prediction = (np.mean(probabilities, axis=0) >= config["threshold"]).astype(np.int8)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / NAME
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["id", "prediction"])
        writer.writerows(zip(ids, prediction.tolist()))
    with zipfile.ZipFile(args.output / (NAME + ".zip"), "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(csv_path, arcname=NAME)
    print(json.dumps({"count": len(ids), "threshold": config["threshold"], "test_labels_used": False,
                      "archive": str((args.output / (NAME + ".zip")).resolve())}))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
