"""Rebuild a transparent, independent 434-feature JARVIS train/val package.

Only official train/val labels and the raw database's jid/atoms are consumed.
The inaccessible predecessor feature package is NOT reproduced byte-for-byte:
geometry definitions below are explicit and their unknown equivalence is recorded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import warnings
import zipfile

import ijson
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_split(path):
    """Stop the event stream after train/val; never request a test-label event."""
    result = {"train": {}, "val": {}}
    completed = set()
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".json"))
        with archive.open(name) as stream:
            for prefix, event, value in ijson.parse(stream):
                if prefix == "" and event == "map_key" and value == "test":
                    if completed != {"train", "val"}:
                        raise ValueError("Test precedes train/val; refusing to parse test labels")
                    break
                field, sep, jid = prefix.partition(".")
                if field in result and sep and event in ("number", "string", "boolean"):
                    if value not in (0, 1):
                        raise ValueError("Official label is not binary")
                    result[field][jid] = int(value)
                if prefix in result and event == "end_map":
                    completed.add(prefix)
                    if completed == {"train", "val"}:
                        break
    if completed != {"train", "val"}:
        raise ValueError("Missing official train/val split")
    return result


def read_atoms(path, selected):
    """Decode one raw record at a time; retain only selected jid/atoms.

    Some irrelevant raw fields contain nonstandard NaN/Infinity tokens. The
    stdlib decoder accepts these (as None here), unlike ijson's strict parser.
    No raw DFT target field is inspected or retained.
    """
    output = {}
    decoder = json.JSONDecoder(parse_constant=lambda _: None)
    with Path(path).open("r", encoding="utf-8") as stream:
        buffer = stream.read(1024 * 1024)
        if not buffer.startswith("["):
            raise ValueError("Raw database is not a JSON array")
        position = 1
        while True:
            while position < len(buffer) and buffer[position] in " \r\n\t,":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                break
            try:
                record, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                remainder = buffer[position:]
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    raise ValueError("Truncated or invalid raw database")
                buffer, position = remainder + chunk, 0
                continue
            position = end
            jid = record["jid"]
            if jid in selected:
                raw = record["atoms"]
                atoms = {key: raw[key] for key in ("lattice_mat", "coords", "elements", "cartesian")}
                if jid in output:
                    raise ValueError(f"Duplicate raw jid: {jid}")
                output[jid] = atoms
                if len(output) == len(selected):
                    break
    if missing := selected - output.keys():
        raise ValueError(f"Raw database missing {len(missing)} train/val structures")
    return output


_featurizers = None
_composition_cache = {}
_element_cache = {}


def initialize_worker():
    global _featurizers
    warnings.filterwarnings("ignore")
    from matminer.featurizers.composition import ElementProperty, IonProperty, Stoichiometry, ValenceOrbital
    _featurizers = [Stoichiometry(), ElementProperty.from_preset("magpie"),
                    ValenceOrbital(props=["frac"]), IonProperty(fast=True)]


def element_info(symbol):
    if symbol not in _element_cache:
        from pymatgen.core import Element
        element = Element(symbol)
        _element_cache[symbol] = (element.Z, element.block in ("d", "f"), float(element.X or np.nan))
    return _element_cache[symbol]


def composition_features(elements):
    from pymatgen.core import Composition
    counts = Counter(elements)
    divisor = math.gcd(*counts.values())
    key = tuple(sorted((symbol, count // divisor) for symbol, count in counts.items()))
    if key not in _composition_cache:
        composition = Composition(dict(key))
        parts = []
        failures = []
        for f in _featurizers:
            try:
                parts.extend(f.featurize(composition))
            except Exception as exc:
                parts.extend([np.nan] * len(f.feature_labels()))
                failures.append(type(f).__name__ + ": " + type(exc).__name__)
        if len(parts) != 145:
            raise ValueError(f"Matminer composition schema changed: {len(parts)} columns")
        _composition_cache[key] = (np.asarray(parts, dtype=float), failures)
    parts, failures = _composition_cache[key]
    fractions = np.zeros(118, dtype=float)
    for symbol, count in counts.items():
        fractions[element_info(symbol)[0] - 1] = count / len(elements)
    nonzero = fractions[fractions > 0]
    summary = [len(counts), float(-np.sum(nonzero * np.log(nonzero))),
               float(np.linalg.norm(fractions)), float(nonzero.max())]
    return parts, fractions, summary, failures


def statistics(values, empty=0.0):
    if len(values) == 0:
        return [empty] * 4
    values = np.asarray(values)
    return [float(np.mean(values)), float(np.std(values)), float(np.min(values)), float(np.max(values))]


def structure_key(lattice, frac, elements, decimals=5):
    """Translation/site-order invariant within one lattice basis; not symmetry matching."""
    counts = Counter(elements)
    anchor_element = min(counts, key=lambda e: (counts[e], e))
    candidates = []
    for anchor in np.flatnonzero(np.asarray(elements) == anchor_element):
        shifted = np.mod(frac - frac[anchor], 1.0)
        shifted = np.mod(np.round(shifted, decimals), 1.0)
        sites = sorted((symbol, *xyz.tolist()) for symbol, xyz in zip(elements, shifted))
        candidates.append(sites)
    canonical_lattice = np.round(lattice, decimals)
    canonical_lattice[np.abs(canonical_lattice) < 10 ** -decimals] = 0.0
    payload = [canonical_lattice.tolist(), min(candidates)]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def compute_one(item):
    jid, atoms = item
    from pymatgen.core import Structure
    lattice = np.asarray(atoms["lattice_mat"], dtype=float)
    coords = np.asarray(atoms["coords"], dtype=float)
    elements = atoms["elements"]
    n = len(elements)
    if n == 0 or not np.isfinite(lattice).all() or not np.isfinite(coords).all() or abs(np.linalg.det(lattice)) < 1e-8:
        raise ValueError(f"Degenerate structure: {jid}")
    frac = coords @ np.linalg.inv(lattice) if atoms["cartesian"] else coords
    frac = np.mod(frac, 1.0)
    structure = Structure(lattice, elements, frac, coords_are_cartesian=False, to_unit_cell=True)
    comp, fractions, summary, failures = composition_features(elements)
    lengths = np.sort(np.linalg.norm(lattice, axis=1))
    cosines = sorted(float(np.dot(lattice[i], lattice[j]) /
                           (np.linalg.norm(lattice[i]) * np.linalg.norm(lattice[j])))
                     for i, j in ((0, 1), (0, 2), (1, 2)))
    simple = summary + [n, abs(float(np.linalg.det(lattice))) / n,
                        lengths[0] / lengths[2], lengths[1] / lengths[2], *cosines]
    centers, neighbors, _, distances = structure.get_neighbor_list(6.0 + 1e-8, numerical_tol=1e-8,
                                                                  exclude_self=True)
    valid = (distances > 1e-6) & (distances <= 6.0 + 1e-8)
    centers, neighbors, distances = centers[valid], neighbors[valid], distances[valid]
    info = [element_info(symbol) for symbol in elements]
    df = np.asarray([value[1] for value in info])
    en = np.asarray([value[2] for value in info])
    masks = [np.ones(len(distances), dtype=bool), df[centers], df[centers] & df[neighbors]]
    center_counts = [n, int(df.sum()), int(df.sum())]
    geometry = []
    bins = np.arange(0, 6.0001, .25)
    for mask, denominator in zip(masks, center_counts):
        histogram = np.histogram(np.minimum(distances[mask], 6.0), bins=bins)[0].astype(float)
        geometry.extend((histogram / max(denominator, 1)).tolist())
    for radius in (2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
        for i, mask in enumerate(masks):
            counts = np.bincount(centers[mask & (distances <= radius)], minlength=n)
            geometry.extend(statistics(counts if i == 0 else counts[df]))
    # nearest_df uses d/f centers and d/f neighbors, bounded at the same 6 A cutoff.
    for mask, allowed in ((masks[0], np.ones(n, dtype=bool)), (masks[2], df)):
        nearest = np.full(n, 6.0, dtype=float)
        np.minimum.at(nearest, centers[mask], distances[mask])
        geometry.extend(statistics(nearest[allowed], empty=0.0))
    for mask in (masks[0], masks[2]):
        differences = np.abs(en[centers[mask]] - en[neighbors[mask]])
        differences = differences[np.isfinite(differences)]
        geometry.extend(statistics(differences))
    features = np.concatenate((comp, fractions, simple, geometry)).astype(np.float32)
    if len(features) != 434 or np.isinf(features).any():
        raise ValueError(f"Invalid feature vector: {jid}, {len(features)}")
    return jid, features, structure_key(lattice, frac, elements), failures


def union_groups(ids, hashes, reference_groups):
    parent = {jid: jid for jid in ids}

    def find(jid):
        while parent[jid] != jid:
            parent[jid] = parent[parent[jid]]
            jid = parent[jid]
        return jid

    def merge(group):
        group = [jid for jid in group if jid in parent]
        if group:
            root = find(group[0])
            for jid in group[1:]:
                parent[find(jid)] = root

    by_hash = defaultdict(list)
    for jid in ids:
        by_hash[hashes[jid]].append(jid)
    for group in list(by_hash.values()) + reference_groups:
        merge(group)
    final = defaultdict(list)
    for jid in ids:
        final[find(jid)].append(jid)
    return [group for group in final.values() if len(group) > 1], [group for group in by_hash.values() if len(group) > 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="predecessor output experiment.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="max rows of each split for smoke test")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    names = reference["data_manifest"]["feature_names"]
    if len(names) != 434:
        raise ValueError("Expected predecessor schema of 434 columns")
    split = read_split(args.split)
    excluded = sorted(set(split["train"]) & set(split["val"]))
    for jid in excluded:
        del split["val"][jid]
    full_counts = {key: len(value) for key, value in split.items()}
    if args.limit:
        split = {key: dict(list(value.items())[:args.limit]) for key, value in split.items()}
    selected = set(split["train"]) | set(split["val"])
    print(json.dumps({"phase": "read", "counts": {k: len(v) for k, v in split.items()},
                      "excluded_duplicate_val_ids": excluded, "test_accessed": False}), flush=True)
    atoms = read_atoms(args.raw, selected)
    rows, hashes, errors = {}, {}, {}
    initialize_worker()
    labels = [name for featurizer in _featurizers for name in featurizer.feature_labels()]
    if labels != names[:145]:
        raise ValueError("Installed Matminer feature labels do not match reference")
    items = list(atoms.items())
    if args.workers == 1:
        iterator = map(compute_one, items)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker)
        iterator = pool.map(compute_one, items, chunksize=128)
    try:
        for count, (jid, features, structure_hash, failures) in enumerate(iterator, 1):
            rows[jid], hashes[jid] = features, structure_hash
            if failures:
                errors[jid] = failures
            if count % 1000 == 0 or count == len(items):
                print(json.dumps({"phase": "features", "done": count, "total": len(items),
                                  "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    del atoms, items
    ref_groups = reference["config"].get("duplicate_groups", [])
    groups, rebuilt_groups = union_groups(list(split["train"]), hashes, ref_groups)
    train_hashes = {hashes[jid] for jid in split["train"]}
    overlap_val = [jid for jid in split["val"] if hashes[jid] in train_hashes]
    counts, nan_counts = {}, {}
    for name, labels in split.items():
        ids = np.asarray(list(labels))
        x = np.asarray([rows[jid] for jid in ids], dtype=np.float32)
        y = np.asarray(list(labels.values()), dtype=np.int8)
        payload = {"ids": ids, "X": x, "y": y}
        if name == "val":
            payload["novel_structure"] = np.asarray([hashes[jid] not in train_hashes for jid in ids], dtype=bool)
        np.savez_compressed(args.output / (name + ".npz"), **payload)
        counts[name] = len(ids)
        nan_counts[name] = {"rows": int(np.isnan(x).any(axis=1).sum()),
                            "values": int(np.isnan(x).sum())}
    audit = {"grouping_method": "lattice/periodic fractional sites rounded to 5 decimals, site-order and origin invariant",
             "limitation": "not a full symmetry/primitive-cell equivalence test; distinct lattice bases may be missed",
             "excluded_training_duplicate_val_ids": excluded,
             "reconstructed_train_duplicate_groups": rebuilt_groups,
             "reference_train_duplicate_groups_used": ref_groups,
             "duplicate_groups": groups, "validation_structure_overlap_ids": overlap_val,
             "train_duplicate_groups_with_conflicting_labels": [group for group in groups
                  if len({split["train"][jid] for jid in group}) > 1],
             "test_labels_accessed": False}
    (args.output / "structure_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (args.output / "group_config.json").write_text(json.dumps({"duplicate_groups": groups, "test_included": False}, indent=2) + "\n", encoding="utf-8")
    (args.output / "duplicate_groups.json").write_text(json.dumps({"duplicate_groups": groups, "test_included": False}, indent=2) + "\n", encoding="utf-8")
    known_differences = [
        "Independent reconstruction: predecessor feature data/preparation source was inaccessible; no claim of bitwise equivalence.",
        "Matminer 145 uses Stoichiometry, Magpie elemental properties, fractional ValenceOrbital and fast IonProperty; elemental reference GSmagmom is not the target compound's DFT magmom.",
        "Geometry uses all periodic neighbors to 6 A; rdf is directed pair counts divided by the eligible center count, without shell-volume normalization.",
        "df means elements in periodic-table d/f blocks, including full d shells; df_pair restricts both endpoints.",
        "nearest_df and pauling_diff_df restrict both endpoints to d/f elements; missing nearest neighbors use 6 A, empty populations use zero.",
        "Coordinate rounds/origin invariance may differ from the predecessor audit; new mask and group union are fully recorded.",
    ]
    manifest = {"benchmark": "AI/SinglePropertyClass/dft_3d_magmom_oszicar",
                "feature_names": names, "feature_count": 434, "split_counts": counts,
                "group_config_path": "group_config.json", "duplicate_groups_path": "duplicate_groups.json",
                "full_split_counts": full_counts, "test_included": False,
                "validation_excludes_training_duplicate_id": True,
                "novel_validation_mask": audit["grouping_method"],
                "novel_validation_count": counts["val"] - len(overlap_val),
                "test_labels_accessed": False, "raw_fields_used": ["jid", "atoms.lattice_mat", "atoms.coords", "atoms.elements", "atoms.cartesian"],
                "reference_experiment_sha256": digest(args.reference), "raw_file_sha256": digest(args.raw),
                "split_zip_sha256": digest(args.split), "script_sha256": digest(__file__),
                "structure_audit_sha256": digest(args.output / "structure_audit.json"),
                "file_sha256": {name: digest(args.output / name) for name in ("train.npz", "val.npz")},
                "known_differences": known_differences, "nan_counts": nan_counts,
                "featurizer_failures": errors, "partial_smoke_test": bool(args.limit),
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "matminer", "pymatgen", "ijson")},
                "seconds": time.monotonic() - started}
    (args.output / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"phase": "complete", "counts": counts, "features": 434,
                      "novel_validation_count": manifest["novel_validation_count"],
                      "train_duplicate_groups": len(groups), "nan_counts": nan_counts,
                      "featurizer_failures": len(errors), "seconds": manifest["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
