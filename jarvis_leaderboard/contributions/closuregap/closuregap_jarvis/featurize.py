from __future__ import annotations

import math
import re


ELEMENTS = {
    "H": (1, 2.20, 53), "C": (6, 2.55, 67), "N": (7, 3.04, 56), "O": (8, 3.44, 48),
    "Na": (11, 0.93, 190), "Mg": (12, 1.31, 145), "Al": (13, 1.61, 118),
    "Si": (14, 1.90, 111), "P": (15, 2.19, 98), "S": (16, 2.58, 88),
    "Cl": (17, 3.16, 79), "K": (19, 0.82, 243), "Ca": (20, 1.00, 194),
    "Ti": (22, 1.54, 176), "Fe": (26, 1.83, 156), "Cu": (29, 1.90, 145),
    "Zn": (30, 1.65, 142), "Ga": (31, 1.81, 136), "Ge": (32, 2.01, 125),
    "As": (33, 2.18, 114), "Se": (34, 2.55, 103), "Br": (35, 2.96, 94),
    "Sr": (38, 0.95, 219), "In": (49, 1.78, 156), "Sn": (50, 1.96, 145),
    "I": (53, 2.66, 115), "Ba": (56, 0.89, 253), "Pb": (82, 2.33, 154),
}


def parse_formula(formula: str) -> dict[str, float]:
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    if not parts:
        raise ValueError(f"could not parse formula: {formula}")
    comp: dict[str, float] = {}
    for elem, count_text in parts:
        if elem not in ELEMENTS:
            raise KeyError(f"element not in embedded table: {elem}")
        count = float(count_text) if count_text else 1.0
        comp[elem] = comp.get(elem, 0.0) + count
    return comp


def composition_features(formula: str) -> list[float]:
    comp = parse_formula(formula)
    total = sum(comp.values())
    fracs = {e: c / total for e, c in comp.items()}
    z = sum(fracs[e] * ELEMENTS[e][0] for e in fracs)
    en = sum(fracs[e] * ELEMENTS[e][1] for e in fracs)
    radius = sum(fracs[e] * ELEMENTS[e][2] for e in fracs)
    entropy = -sum(f * math.log(max(f, 1e-12)) for f in fracs.values())
    max_en_gap = max(ELEMENTS[e][1] for e in fracs) - min(ELEMENTS[e][1] for e in fracs)
    return [1.0, z, en, radius, entropy, max_en_gap, len(fracs), total]
