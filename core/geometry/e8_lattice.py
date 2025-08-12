import numpy as np
from itertools import permutations, product

class E8Lattice:
    def __init__(self):
        self.dimension = 8
        self.roots = self._generate_roots()

    def _generate_roots(self):
        roots = []
        # Type 1: permutations of (±1, ±1, 0, ..., 0)
        for pos in permutations(range(8), 2):
            for signs in product([-1, 1], repeat=2):
                vec = np.zeros(8)
                vec[list(pos)] = signs
                roots.append(vec)
        # Type 2: (±0.5)^8 with even number of minus signs
        for signs in product([-1, 1], repeat=8):
            if np.sum(np.array(signs) < 0) % 2 == 0:
                vec = np.array(signs) * 0.5
                roots.append(vec)
        return np.array(roots)  # Shape: (240, 8)

    def is_on_lattice(self, point):
        p = np.array(point)
        if p.shape != (8,):
            return False
        # Check if all integers or all half-integers
        floors = np.floor(p)
        is_int = np.all(np.abs(p - floors) < 1e-10)
        is_half = np.all(np.abs(p - (floors + 0.5)) < 1e-10)
        if not (is_int or is_half):
            return False
        # Check if sum is even
        s = np.sum(p)
        return np.abs(s - 2 * np.round(s / 2)) < 1e-10

    def project_to_lattice(self, vector):
        v = np.array(vector)
        if v.shape != (8,):
            raise ValueError("Vector must be 8-dimensional")

        def closest_in_dn(x, n=8):
            r = np.round(x).astype(float)
            errs = x - r
            sum_r = np.sum(r)
            if np.abs(sum_r - 2 * np.round(sum_r / 2)) < 1e-10:
                return r
            # Adjust one coordinate by ±1 to make sum even, minimizing distance increase
            costs = []
            adjustments = []
            for i in range(n):
                # +1 at i
                cost_plus = (-2 * errs[i] + 1)
                costs.append(cost_plus)
                adjustments.append((i, 1.0))
                # -1 at i
                cost_minus = (2 * errs[i] + 1)
                costs.append(cost_minus)
                adjustments.append((i, -1.0))
            min_idx = np.argmin(costs)
            adj_i, adj_dir = adjustments[min_idx]
            new_r = r.copy()
            new_r[adj_i] += adj_dir
            return new_r

        # Closest in D8 (even sum integers)
        closest_int = closest_in_dn(v)
        dist_int = np.sum((v - closest_int)**2)
        # Closest in D8 + (0.5)^8
        half = np.full(8, 0.5)
        closest_half = closest_in_dn(v - half) + half
        dist_half = np.sum((v - closest_half)**2)
        if dist_int <= dist_half:
            return closest_int
        else:
            return closest_half

    def get_neighbors(self, point):
        p = np.array(point)
        if not self.is_on_lattice(p):
            raise ValueError("Point not on lattice")
        return p + self.roots  # Shape: (240, 8)

    def lattice_distance(self, p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        if not (self.is_on_lattice(p1) and self.is_on_lattice(p2)):
            raise ValueError("Points not on lattice")
        diff = p1 - p2
        return np.sqrt(np.sum(diff**2))

    def compute_casimir_invariants(self, point):
        """
        Compute the 8 conserved quantities using power sums at the degrees of E8 invariants
        (2,8,12,14,18,20,24,30) as proxies for the actual polynomial invariants.
        """
        p = np.array(point)
        if not self.is_on_lattice(p):
            raise ValueError("Point not on lattice")
        degrees = [2, 8, 12, 14, 18, 20, 24, 30]
        return np.array([np.sum(p**d) for d in degrees])

    def quantization_error(self, vector):
        proj = self.project_to_lattice(vector)
        return np.linalg.norm(vector - proj)

    def snap_back(self, vector):
        return self.project_to_lattice(vector)