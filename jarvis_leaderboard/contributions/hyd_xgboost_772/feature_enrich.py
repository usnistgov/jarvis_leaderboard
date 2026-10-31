"""Append atom-only magnetic chemistry and adaptive local-environment features.

The original 434 columns and all split metadata are preserved. Only raw jid/atoms
are retained; test labels and computed compound DFT properties are never used.
All element priors below are fixed periodic-table descriptors, not fitted targets.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import warnings

import numpy as np


BASE_SCRIPT = Path(__file__).resolve().with_name("base_prepare_features.py")
if not BASE_SCRIPT.exists():
    BASE_SCRIPT = Path(__file__).resolve().parents[2] / "member2912_20261030" / "src" / "prepare_features.py"
spec = importlib.util.spec_from_file_location("old_features", BASE_SCRIPT)
old_features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_features)

MAG3D = frozenset("Ti V Cr Mn Fe Co Ni Cu".split())
MAG4D = frozenset("Zr Nb Mo Tc Ru Rh Pd Ag".split())
MAG5D = frozenset("Hf Ta W Re Os Ir Pt Au".split())
LANTHANIDE = frozenset("Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb".split())
ACTINIDE = frozenset("Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No".split())
FOPEN = LANTHANIDE | ACTINIDE
ACTIVE = MAG3D | MAG4D | MAG5D | FOPEN
ANION = frozenset("B C N O F Si P S Cl Ge As Se Br Sb Te I".split())
HALOGEN = frozenset("F Cl Br I".split())
CENTER_GROUPS = ("all", "mag3d", "fopen", "Fe", "Mn", "Co", "Ni", "Cr")
NEIGHBOR_GROUPS = ("all", "active", "anion", "O")
STAT_NAMES = ("mean", "std", "min", "max")
PRIOR_NAMES = ("unpaired_total", "unpaired_d", "unpaired_f", "unpaired_df", "holes_d", "holes_f", "common_ox_min", "common_ox_max")


def initialize_worker():
    warnings.filterwarnings("ignore")


@lru_cache(maxsize=None)
def element_prior(symbol):
    from pymatgen.core import Element
    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
    element = Element(symbol)
    capacities = {"s": 2, "p": 6, "d": 10, "f": 14}
    unpaired = {shell: 0 for shell in capacities}
    highest = {shell: 0 for shell in capacities}
    for principal, shell, number in element.full_electronic_structure:
        unpaired[shell] += min(number, capacities[shell] - number)
        highest[shell] = number
    ox = element.common_oxidation_states
    props = [sum(unpaired.values()), unpaired["d"], unpaired["f"], unpaired["d"] + unpaired["f"],
             10 - highest["d"] if highest["d"] else 0,
             14 - highest["f"] if highest["f"] else 0,
             min(ox) if ox else 0, max(ox) if ox else 0]
    radius = float(CovalentRadius.radius.get(symbol, element.atomic_radius or 1.5))
    electronegativity = float(element.X or 0.0)
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.5
    if not np.isfinite(electronegativity):
        electronegativity = 0.0
    return np.array(props, dtype=float), radius, electronegativity, float(element.atomic_mass)


def stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [0.0] * 4
    return [float(values.mean()), float(values.std()), float(values.min()), float(values.max())]


def mask_for(symbols, group):
    if group == "all":
        return np.ones(len(symbols), dtype=bool)
    members = {"mag3d": MAG3D, "fopen": FOPEN, "active": ACTIVE, "anion": ANION}.get(group, {group})
    return np.array([symbol in members for symbol in symbols], dtype=bool)


def extra_names():
    names = []
    for prior in PRIOR_NAMES:
        names.extend(f"atom_prior_{prior}_{stat}" for stat in STAT_NAMES)
    for group in ("mag3d", "mag4d", "mag5d", "lanthanide", "actinide", "anion", "halogen", "O", "N", "S", "closed_d", "active"):
        names.append(f"chem_fraction_{group}")
    names.extend(("geom_density_g_cm3", "geom_covalent_sphere_volume_fraction", "geom_volume_over_covalent_mean_cubed",
                  "geom_lattice_length_std_over_mean", "geom_lattice_angle_deviation_90", "chem_unpaired_df_density"))
    for center in CENTER_GROUPS:
        for neighbor in NEIGHBOR_GROUPS:
            names.extend(f"local_{center}_nearest_{neighbor}_{stat}" for stat in STAT_NAMES)
        names.extend(f"local_{center}_nearest_radius_normalized_{stat}" for stat in STAT_NAMES)
        for ratio in (1.15, 1.35):
            names.extend(f"shell_{center}_r{ratio:g}_coord_{stat}" for stat in STAT_NAMES)
        for prop in ("O_fraction", "N_fraction", "halogen_fraction", "anion_fraction", "active_fraction", "en_difference", "neighbor_df_unpaired", "center_neighbor_df_product"):
            names.append(f"shell_{center}_r1.35_{prop}_mean")
    return names


def compute_one(item):
    from pymatgen.core import Structure
    jid, atoms = item
    lattice = np.array(atoms["lattice_mat"], dtype=float)
    coords = np.array(atoms["coords"], dtype=float)
    symbols = atoms["elements"]
    n = len(symbols)
    frac = coords @ np.linalg.inv(lattice) if atoms["cartesian"] else coords
    structure = Structure(lattice, symbols, np.mod(frac, 1.0), to_unit_cell=True)
    properties = [element_prior(symbol) for symbol in symbols]
    priors = np.stack([p[0] for p in properties])
    radii = np.array([p[1] for p in properties])
    en = np.array([p[2] for p in properties])
    masses = np.array([p[3] for p in properties])
    features = []
    for col in priors.T:
        features.extend(stats(col))
    groups = {"mag3d": MAG3D, "mag4d": MAG4D, "mag5d": MAG5D, "lanthanide": LANTHANIDE, "actinide": ACTINIDE,
              "anion": ANION, "halogen": HALOGEN, "O": {"O"}, "N": {"N"}, "S": {"S"},
              "closed_d": {"Zn", "Cd", "Hg"}, "active": ACTIVE}
    for group in groups.values():
        features.append(sum(symbol in group for symbol in symbols) / n)
    volume = abs(float(np.linalg.det(lattice)))
    lengths = np.linalg.norm(lattice, axis=1)
    angles = [math.degrees(math.acos(float(np.clip(np.dot(lattice[i], lattice[j]) / (lengths[i] * lengths[j]), -1, 1))))
              for i, j in ((0, 1), (0, 2), (1, 2))]
    features.extend((masses.sum() / volume * 1.6605390666, np.sum(4 / 3 * np.pi * radii ** 3) / volume,
                     volume / n / (radii.mean() ** 3), lengths.std() / lengths.mean(),
                     float(np.mean(np.abs(np.asarray(angles) - 90))), priors[:, 3].sum() / volume))
    centers, neighbors, _, distances = structure.get_neighbor_list(6.0 + 1e-8, numerical_tol=1e-8, exclude_self=True)
    valid = (distances > 1e-6) & (distances <= 6.0 + 1e-8)
    centers, neighbors, distances = centers[valid], neighbors[valid], distances[valid]
    nearest = np.full(n, 6.0)
    np.minimum.at(nearest, centers, distances)
    norm_distance = distances / (radii[centers] + radii[neighbors])
    nearest_norm = np.full(n, 6.0)
    np.minimum.at(nearest_norm, centers, norm_distance)
    neighbor_masks = {g: mask_for(symbols, g) for g in NEIGHBOR_GROUPS}
    shell_masks = {ratio: distances <= nearest[centers] * ratio + 1e-8 for ratio in (1.15, 1.35)}
    shell_counts = {ratio: np.bincount(centers[mask], minlength=n) for ratio, mask in shell_masks.items()}
    shell = shell_masks[1.35]
    shell_props = [np.array([s == "O" for s in symbols], float)[neighbors],
                   np.array([s == "N" for s in symbols], float)[neighbors],
                   np.array([s in HALOGEN for s in symbols], float)[neighbors],
                   neighbor_masks["anion"][neighbors].astype(float),
                   neighbor_masks["active"][neighbors].astype(float),
                   np.abs(en[centers] - en[neighbors]), priors[neighbors, 3],
                   priors[centers, 3] * priors[neighbors, 3]]
    per_site_props = [np.bincount(centers[shell], weights=value[shell], minlength=n) / np.maximum(shell_counts[1.35], 1)
                      for value in shell_props]
    for group in CENTER_GROUPS:
        selected = mask_for(symbols, group)
        for neighbor_group in NEIGHBOR_GROUPS:
            allowed = neighbor_masks[neighbor_group][neighbors]
            nearest_for_group = np.full(n, 6.0)
            np.minimum.at(nearest_for_group, centers[allowed], distances[allowed])
            features.extend(stats(nearest_for_group[selected]))
        features.extend(stats(nearest_norm[selected]))
        for ratio in (1.15, 1.35):
            features.extend(stats(shell_counts[ratio][selected]))
        for values in per_site_props:
            features.append(float(values[selected].mean()) if selected.any() else 0.0)
    features = np.asarray(features, dtype=np.float32)
    if len(features) != len(extra_names()) or not np.isfinite(features).all():
        raise ValueError(f"Invalid feature row: {jid}, columns {len(features)}")
    return jid, features


def featurize_atoms(atoms, jid="inference"):
    """Return the full 772-column vector for one jid/atoms-only input.

    Accepts lattice_mat, coords, elements, and cartesian, matching the raw JARVIS
    atom schema. Requires MatMiner/Pymatgen versions from the package manifest.
    """
    if old_features._featurizers is None:
        old_features.initialize_worker()
    base = old_features.compute_one((jid, atoms))[1]
    extra = compute_one((jid, atoms))[1]
    return np.concatenate((base, extra)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    base_manifest = json.loads((args.base / "data_manifest.json").read_text(encoding="utf-8"))
    if base_manifest["test_included"] or base_manifest["feature_count"] != 434:
        raise ValueError("Unexpected base feature package")
    packages = {}
    for split in ("train", "val"):
        with np.load(args.base / f"{split}.npz", allow_pickle=False) as data:
            packages[split] = {key: data[key][:args.limit] if args.limit else data[key] for key in data.files}
    selected = {str(jid) for data in packages.values() for jid in data["ids"]}
    print(json.dumps({"phase": "read_atoms", "selected": len(selected), "test_labels_accessed": False}), flush=True)
    atoms = old_features.read_atoms(args.raw, selected)
    rows = {}
    initialize_worker()
    if args.workers == 1:
        iterator = map(compute_one, atoms.items())
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker)
        iterator = pool.map(compute_one, atoms.items(), chunksize=64)
    try:
        for done, (jid, values) in enumerate(iterator, 1):
            rows[jid] = values
            if done % 1000 == 0 or done == len(atoms):
                print(json.dumps({"phase": "features", "done": done, "total": len(atoms), "seconds": round(time.monotonic() - started, 1)}), flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    names = base_manifest["feature_names"] + extra_names()
    assert len(names) == len(set(names))
    for split, data in packages.items():
        data["X"] = np.column_stack((data["X"], np.stack([rows[str(jid)] for jid in data["ids"]]))).astype(np.float32)
        np.savez_compressed(args.output / f"{split}.npz", **data)
    for file in ("duplicate_groups.json", "group_config.json", "structure_audit.json"):
        shutil.copy2(args.base / file, args.output / file)
    shutil.copy2(Path(__file__), args.output / "feature_enrich.py")
    shutil.copy2(BASE_SCRIPT, args.output / "base_prepare_features.py")
    manifest = {**base_manifest, "feature_names": names, "feature_count": len(names), "base_feature_count": 434,
                "extra_feature_count": len(extra_names()), "base_data_manifest_sha256": old_features.digest(args.base / "data_manifest.json"),
                "base_package_path": str(args.base.resolve()), "extra_script_sha256": old_features.digest(__file__),
                "base_script_sha256": old_features.digest(BASE_SCRIPT),
                "file_sha256": {file: old_features.digest(args.output / file) for file in ("train.npz", "val.npz")},
                "split_counts": {split: len(data["ids"]) for split, data in packages.items()},
                "partial_smoke_test": bool(args.limit), "test_included": False, "test_labels_accessed": False,
                "extra_feature_definition": {"atom_priors": "Fixed neutral-atom shell occupancy/Hund unpaired counts, common oxidation extrema; not measured material magnetism",
                  "active_element_groups": {key: sorted(value) for key, value in {"mag3d": MAG3D, "mag4d": MAG4D, "mag5d": MAG5D, "fopen": FOPEN, "anion": ANION}.items()},
                  "neighbor_cutoff_angstrom": 6.0, "missing_neighbor_distance_angstrom": 6.0,
                  "adaptive_shell_radii": "1.15 and 1.35 times each center's nearest distance, capped at the 6 A neighbor list",
                  "radius_normalization": "Cordero covalent-radius sum (fallback atomic radius or 1.5 A)",
                  "missing_center_statistics": 0.0},
                "seconds": time.monotonic() - started}
    (args.output / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"phase": "complete", "features": len(names), "extra_features": len(extra_names()),
                      "counts": manifest["split_counts"], "seconds": manifest["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
