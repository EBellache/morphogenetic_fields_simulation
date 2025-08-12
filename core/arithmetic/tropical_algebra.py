import numpy as np
from typing import List, Tuple


class TropicalOperations:
    """
    Implements max-plus (tropical) algebra operations for morphogenetic optimization.

    As described in Chapter 12 of the manuscript, tropical algebra provides a framework
    for projecting analytic functions onto piecewise-linear structures, enabling
    efficient computation of optimal paths in developmental landscapes. This is
    particularly useful for modeling viscous flow constraints and defect-mediated
    decision-making in low-Reynolds-number environments.
    """

    @staticmethod
    def tropical_add(a: float, b: float) -> float:
        """
        Tropical addition: a ⊕ b = max(a, b).

        Biologically: Represents selection of the dominant signal or constraint
        in multi-factorial developmental decisions.
        """
        return max(a, b)

    @staticmethod
    def tropical_multiply(a: float, b: float) -> float:
        """
        Tropical multiplication: a ⊗ b = a + b.

        Biologically: Accumulates costs or energies along developmental paths,
        as in constructal network optimization.
        """
        return a + b

    @staticmethod
    def tropical_power(a: float, n: int) -> float:
        """
        Tropical power: a^⊗n = n * a (repeated tropical multiplication).

        Biologically: Scales signals with hierarchy level in p-adic structures.
        """
        if n < 0:
            raise ValueError("Exponent must be non-negative for tropical power")
        return n * a

    @staticmethod
    def tropical_inner_product(x: np.ndarray, y: np.ndarray) -> float:
        """
        Tropical inner product: max_i (x_i + y_i).

        Biologically: Computes maximal coupling between state vectors in E8 phase space,
        used for defect-field interactions.
        """
        if x.shape != y.shape:
            raise ValueError("Vectors must have the same shape")
        return np.max(x + y)

    @staticmethod
    def tropical_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Tropical matrix multiplication: (A ⊗ B)_ij = max_k (A_ik + B_kj).

        Used for max-plus Viterbi-like decoding in neural spike data analysis
        and optimal branching in constructal networks (manuscript Section 1.3).

        Implemented with NumPy broadcasting for efficiency.
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError("Incompatible matrix shapes for multiplication")
        # Broadcast: A (m,k,1) + B (1,k,n) -> (m,k,n), max over k (axis=1)
        C = np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)
        return C

    @staticmethod
    def newton_polygon(coefficients: List[Tuple[int, float]]) -> List[Tuple[float, float]]:
        """
        Computes the Newton polygon for polynomial coefficients, used for spike time prediction.

        From manuscript: corners of the polygon predict transition points (spike times)
        in piecewise-linear approximations of bioelectric signals.

        Args:
            coefficients: List of (degree n, coefficient a) tuples.

        Returns:
            List of (n, log|a|) points forming the lower convex hull.
        """
        # Filter non-zero coefficients and compute (n, log|a|)
        points = [(float(n), np.log(abs(a))) for n, a in coefficients if a != 0]
        if len(points) < 2:
            return points

        # Sort by degree (x-coordinate)
        points.sort(key=lambda p: p[0])

        # Build lower convex hull (Graham scan variant)
        hull = []
        for p in points:
            while len(hull) >= 2:
                # Vectors: from hull[-2] to hull[-1], from hull[-1] to p
                v1 = (hull[-1][0] - hull[-2][0], hull[-1][1] - hull[-2][1])
                v2 = (p[0] - hull[-1][0], p[1] - hull[-1][1])
                # Cross product v1 x v2 = v1[0]*v2[1] - v1[1]*v2[0]
                # For lower hull, pop if cross <= 0 (not a right turn, i.e., concave or collinear)
                if v1[0] * v2[1] - v1[1] * v2[0] <= 0:
                    hull.pop()
                else:
                    break
            hull.append(p)

        return hull