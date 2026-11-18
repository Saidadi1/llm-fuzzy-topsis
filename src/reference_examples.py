"""Published numerical example used to validate the from-scratch fuzzy TOPSIS.

C.-T. Chen, "Extensions of the TOPSIS for group decision-making under fuzzy
environment", Fuzzy Sets and Systems 114 (2000) 1-9, Section 4.
A software company hires a system analysis engineer: 3 candidates (A1..A3),
3 decision makers (D1..D3), 5 benefit criteria:
C1 emotional steadiness, C2 oral communication skill, C3 personality,
C4 past experience, C5 self-confidence.
Data transcribed from Tables 3 and 4; expected values from Table 8 / Step 8.
"""

from __future__ import annotations

import numpy as np

from .fuzzy import CHEN_RATING_SCALE, CHEN_WEIGHT_SCALE, aggregate

CRITERIA = ["C1 emotional steadiness", "C2 oral communication", "C3 personality",
            "C4 past experience", "C5 self-confidence"]
ALTERNATIVES = ["A1", "A2", "A3"]

# Table 3: importance weights, rows = criteria, columns = D1, D2, D3
WEIGHTS = [
    ["H", "VH", "MH"],
    ["VH", "VH", "VH"],
    ["VH", "H", "H"],
    ["VH", "VH", "VH"],
    ["M", "MH", "MH"],
]

# Table 4: ratings[criterion][alternative] = (D1, D2, D3)
RATINGS = [
    [["MG", "G", "MG"], ["G", "G", "MG"], ["VG", "G", "F"]],      # C1
    [["G", "MG", "F"], ["VG", "VG", "VG"], ["MG", "G", "VG"]],    # C2
    [["F", "G", "G"], ["VG", "VG", "G"], ["G", "MG", "VG"]],      # C3
    [["VG", "G", "VG"], ["VG", "VG", "VG"], ["G", "VG", "MG"]],   # C4
    [["F", "F", "F"], ["VG", "MG", "G"], ["G", "G", "MG"]],       # C5
]

# Published results (rounded to 2 decimals in the paper)
EXPECTED_D_PLUS = np.array([2.10, 1.24, 1.59])
EXPECTED_D_MINUS = np.array([3.45, 4.13, 3.85])
EXPECTED_CC = np.array([0.62, 0.77, 0.71])
EXPECTED_RANKING = ["A2", "A3", "A1"]

# Table 5 exactly as printed. NOTE - the paper is internally inconsistent:
# * 5 cells of Table 5 are not the mean of the Table 4 ratings (e.g. A1/C2 =
#   mean(G, MG, F) = (5, 7, 8.67) but (5, 7, 9) is printed); the weights are
#   also rounded up (C1 = mean(H, VH, MH) = (0.7, 0.867, 0.967), printed (0.7, 0.9, 1)).
# * Summing the vertex distances of the printed Table 7 gives d1* = 1.94, not the
#   printed 2.10, so CC1 = 0.64 rather than 0.62 (arithmetic erratum in Table 8).
PRINTED_TABLE5_X = np.array([
    [[5.7, 7.7, 9.3], [5, 7, 9], [5.7, 7.7, 9], [8.33, 9.67, 10], [3, 5, 7]],
    [[6.3, 8.3, 9.7], [9, 10, 10], [8.3, 9.7, 10], [9, 10, 10], [7, 9, 10]],
    [[6.3, 8, 9], [7, 9, 10], [7, 9, 10], [7, 9, 10], [6.3, 8.3, 9.7]],
])
PRINTED_TABLE5_W = np.array([[0.7, 0.9, 1], [0.9, 1, 1], [0.77, 0.93, 1], [0.9, 1, 1], [0.43, 0.63, 0.83]])
# d* recomputed by hand from the printed Table 7 (consistent with Tables 5-7)
RECOMPUTED_D_PLUS = np.array([1.94, 1.26, 1.59])


def chen_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Return the aggregated fuzzy decision matrix (3, 5, 3) and weights (5, 3),
    using Chen's arithmetic-mean aggregation (eqs. 13-14)."""
    # ratings tensor: (K=3 decision makers, m=3 alternatives, n=5 criteria, 3)
    r = np.array([[[CHEN_RATING_SCALE.to_tfn(RATINGS[j][i][k]) for j in range(5)]
                   for i in range(3)] for k in range(3)])
    w = np.array([[CHEN_WEIGHT_SCALE.to_tfn(WEIGHTS[j][k]) for j in range(5)] for k in range(3)])
    return aggregate(r, "mean"), aggregate(w, "mean")
