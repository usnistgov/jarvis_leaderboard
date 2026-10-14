"""Composition and crystal-structure features for JARVIS exfoliation inference."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Structure, Lattice


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        val = float(x)
        if np.isnan(val) or np.isinf(val):
            return default
        return val
    except Exception:
        return default


def weighted_stats(values: List[float], weights: List[float]) -> List[float]:
    v = np.asarray(values, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    if len(v) == 0 or float(w.sum()) <= 0:
        return [0.0] * 6
    w = w / w.sum()
    mean = float(np.sum(w * v))
    var = float(np.sum(w * (v - mean) ** 2))
    std = float(np.sqrt(max(var, 0.0)))
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    vrange = float(vmax - vmin)
    mad = float(np.sum(w * np.abs(v - mean)))
    return [mean, std, vmin, vmax, vrange, mad]


def element_fraction_vector(comp: Composition) -> List[float]:
    frac = comp.fractional_composition
    vec = np.zeros(118, dtype=np.float32)
    for el, amount in frac.get_el_amt_dict().items():
        z = Element(el).Z
        vec[z - 1] = float(amount)
    return vec.tolist()


def element_property_vector(comp: Composition) -> List[float]:
    frac = comp.fractional_composition
    el_amt = frac.get_el_amt_dict()
    elems = [Element(sym) for sym in el_amt.keys()]
    weights = [float(el_amt[e.symbol]) for e in elems]

    prop_extractors = [
        ("Z", lambda e: e.Z),
        ("atomic_mass", lambda e: e.atomic_mass),
        ("row", lambda e: e.row),
        ("group", lambda e: e.group),
        ("X", lambda e: e.X),
        ("mendeleev_no", lambda e: e.mendeleev_no),
        ("atomic_radius", lambda e: e.atomic_radius),
        ("average_ionic_radius", lambda e: e.average_ionic_radius),
        ("common_oxidation_state", lambda e: e.common_oxidation_states[0] if e.common_oxidation_states else None),
    ]

    feats: List[float] = []
    for _, getter in prop_extractors:
        vals = [safe_float(getter(e), 0.0) for e in elems]
        feats.extend(weighted_stats(vals, weights))

    n_elems = len(elems)
    num_atoms_reduced = float(comp.reduced_composition.num_atoms)
    total_atoms = float(comp.num_atoms)
    feats.extend([float(n_elems), num_atoms_reduced, total_atoms])
    return feats


def structure_from_any_atoms_dict(atoms: Dict[str, Any]) -> Structure:
    if "lattice" in atoms and "sites" in atoms:
        return Structure.from_dict(atoms)

    if "lattice_mat" in atoms and "coords" in atoms and "elements" in atoms:
        lattice = Lattice(atoms["lattice_mat"])
        coords = atoms["coords"]
        species = atoms["elements"]
        cartesian = bool(atoms.get("cartesian", False))
        return Structure(lattice=lattice, species=species, coords=coords, coords_are_cartesian=cartesian)

    if "lattice_mat" in atoms and "positions" in atoms and "elements" in atoms:
        lattice = Lattice(atoms["lattice_mat"])
        coords = atoms["positions"]
        species = atoms["elements"]
        cartesian = bool(atoms.get("cartesian", False))
        return Structure(lattice=lattice, species=species, coords=coords, coords_are_cartesian=cartesian)

    raise KeyError(f"Unsupported atoms schema. Keys: {sorted(list(atoms.keys()))[:20]}")


def structure_scalar_vector(structure: Structure) -> List[float]:
    lat = structure.lattice
    n_sites = float(len(structure))
    volume = safe_float(structure.volume, 0.0)
    density = safe_float(structure.density, 0.0)
    volume_per_atom = volume / max(n_sites, 1.0)

    dmat = np.asarray(structure.distance_matrix, dtype=np.float32)
    iu = np.triu_indices_from(dmat, k=1)
    pair_d = dmat[iu]
    if pair_d.size == 0:
        pair_stats = [0.0] * 8
    else:
        pair_stats = [
            safe_float(np.min(pair_d)),
            safe_float(np.max(pair_d)),
            safe_float(np.mean(pair_d)),
            safe_float(np.std(pair_d)),
            safe_float(np.quantile(pair_d, 0.10)),
            safe_float(np.quantile(pair_d, 0.25)),
            safe_float(np.quantile(pair_d, 0.50)),
            safe_float(np.quantile(pair_d, 0.75)),
        ]

    return [
        n_sites,
        volume,
        volume_per_atom,
        density,
        safe_float(lat.a, 0.0),
        safe_float(lat.b, 0.0),
        safe_float(lat.c, 0.0),
        safe_float(lat.alpha, 0.0),
        safe_float(lat.beta, 0.0),
        safe_float(lat.gamma, 0.0),
        safe_float(lat.abc[0] / max(lat.abc[1], 1e-8), 0.0),
        safe_float(lat.abc[1] / max(lat.abc[2], 1e-8), 0.0),
        safe_float(lat.abc[0] / max(lat.abc[2], 1e-8), 0.0),
        *pair_stats,
    ]


def featurize_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Return the 196 features in input-record order, with inference controls.

    Each input record must contain exactly ``jid``, ``formula``, and ``atoms``.
    The returned column order is sample_id, weight, f_000..f_195, split_name.
    """
    feature_columns = [f"f_{i:03d}" for i in range(196)]
    columns = ["sample_id", "weight", *feature_columns, "split_name"]
    rows: List[Dict[str, Any]] = []
    seen_ids = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"jid", "formula", "atoms"}:
            raise ValueError("Each record must contain exactly jid, formula, and atoms")
        if record["jid"] is None:
            raise ValueError("Missing jid")
        sample_id = str(record["jid"])
        if not sample_id.strip() or sample_id in seen_ids:
            raise ValueError("Record identifiers must be nonblank and unique")
        seen_ids.add(sample_id)

        comp = Composition(record["formula"])
        structure = structure_from_any_atoms_dict(record["atoms"])
        feats = (
            element_fraction_vector(comp)
            + element_property_vector(comp)
            + structure_scalar_vector(structure)
        )
        if len(feats) != 196:
            raise ValueError("Expected exactly 196 features")
        row: Dict[str, Any] = {"sample_id": sample_id, "weight": 1.0}
        for i, value in enumerate(feats):
            row[f"f_{i:03d}"] = float(value)
        row["split_name"] = "test"
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)
