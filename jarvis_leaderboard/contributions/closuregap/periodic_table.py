# -*- coding: UTF-8 -*-
""" -----------------------------------------
@Project : jarvis_closure_gap
@Core Libraries : (standard library only)
@Author : Ares (Mingze) ares
@Date : 22/07/2026 - 09:40
@Description: Full periodic-table property table for ClosureGap featurization.
@Annotation: ClosureGap's shipped ELEMENTS table only covers ~29 toy elements,
             so real dft_3d formulas raise KeyError. This module provides the
             same 3-field schema (atomic_number, Pauling_electronegativity,
             empirical_atomic_radius_pm) for the whole periodic table. It does
             NOT change the ClosureGap algorithm; it only extends the property
             lookup so the identical descriptor math runs on real materials.
             Elements without a defined Pauling value (noble gases, etc.) use a
             conservative 0.0 placeholder, which only affects two descriptors
             (mean and range of electronegativity) for those rare cases.
----------------------------------------- """
from __future__ import annotations

# symbol -> (atomic number Z, Pauling electronegativity, empirical atomic radius in pm)
# Radii follow common empirical/Slater tabulations; values are used only as
# smooth composition descriptors, not as precise physical constants.
FULL_ELEMENTS: dict[str, tuple[int, float, int]] = {
    "H": (1, 2.20, 25), "He": (2, 0.00, 120),
    "Li": (3, 0.98, 145), "Be": (4, 1.57, 105), "B": (5, 2.04, 85),
    "C": (6, 2.55, 70), "N": (7, 3.04, 65), "O": (8, 3.44, 60),
    "F": (9, 3.98, 50), "Ne": (10, 0.00, 160),
    "Na": (11, 0.93, 180), "Mg": (12, 1.31, 150), "Al": (13, 1.61, 125),
    "Si": (14, 1.90, 110), "P": (15, 2.19, 100), "S": (16, 2.58, 100),
    "Cl": (17, 3.16, 100), "Ar": (18, 0.00, 71),
    "K": (19, 0.82, 220), "Ca": (20, 1.00, 180),
    "Sc": (21, 1.36, 160), "Ti": (22, 1.54, 140), "V": (23, 1.63, 135),
    "Cr": (24, 1.66, 140), "Mn": (25, 1.55, 140), "Fe": (26, 1.83, 140),
    "Co": (27, 1.88, 135), "Ni": (28, 1.91, 135), "Cu": (29, 1.90, 135),
    "Zn": (30, 1.65, 135), "Ga": (31, 1.81, 130), "Ge": (32, 2.01, 125),
    "As": (33, 2.18, 115), "Se": (34, 2.55, 115), "Br": (35, 2.96, 115),
    "Kr": (36, 3.00, 88),
    "Rb": (37, 0.82, 235), "Sr": (38, 0.95, 200),
    "Y": (39, 1.22, 180), "Zr": (40, 1.33, 155), "Nb": (41, 1.60, 145),
    "Mo": (42, 2.16, 145), "Tc": (43, 1.90, 135), "Ru": (44, 2.20, 130),
    "Rh": (45, 2.28, 135), "Pd": (46, 2.20, 140), "Ag": (47, 1.93, 160),
    "Cd": (48, 1.69, 155), "In": (49, 1.78, 155), "Sn": (50, 1.96, 145),
    "Sb": (51, 2.05, 145), "Te": (52, 2.10, 140), "I": (53, 2.66, 140),
    "Xe": (54, 2.60, 108),
    "Cs": (55, 0.79, 260), "Ba": (56, 0.89, 215),
    "La": (57, 1.10, 195), "Ce": (58, 1.12, 185), "Pr": (59, 1.13, 185),
    "Nd": (60, 1.14, 185), "Pm": (61, 1.13, 185), "Sm": (62, 1.17, 185),
    "Eu": (63, 1.20, 185), "Gd": (64, 1.20, 180), "Tb": (65, 1.10, 175),
    "Dy": (66, 1.22, 175), "Ho": (67, 1.23, 175), "Er": (68, 1.24, 175),
    "Tm": (69, 1.25, 175), "Yb": (70, 1.10, 175), "Lu": (71, 1.27, 175),
    "Hf": (72, 1.30, 155), "Ta": (73, 1.50, 145), "W": (74, 2.36, 135),
    "Re": (75, 1.90, 135), "Os": (76, 2.20, 130), "Ir": (77, 2.20, 135),
    "Pt": (78, 2.28, 135), "Au": (79, 2.54, 135), "Hg": (80, 2.00, 150),
    "Tl": (81, 1.62, 190), "Pb": (82, 2.33, 180), "Bi": (83, 2.02, 160),
    "Po": (84, 2.00, 190), "At": (85, 2.20, 127), "Rn": (86, 2.20, 120),
    "Fr": (87, 0.70, 260), "Ra": (88, 0.90, 215),
    "Ac": (89, 1.10, 195), "Th": (90, 1.30, 180), "Pa": (91, 1.50, 180),
    "U": (92, 1.38, 175), "Np": (93, 1.36, 175), "Pu": (94, 1.28, 175),
    "Am": (95, 1.30, 175), "Cm": (96, 1.30, 176),
}
