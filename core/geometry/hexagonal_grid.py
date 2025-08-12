import numpy as np
from typing import List, Tuple, Set, Any

from numpy import floating


class HexagonalGrid:
    """
    Implements a hexagonal grid system using axial coordinates (q, r).

    This grid serves as the spatial substrate for cell placement, field computations,
    and morphogenetic simulations. It supports rectangular bounds in axial space,
    resulting in a parallelogram-shaped grid visually (suitable for worm-like morphologies).

    Coordinates:
    - q: column (axial)
    - r: row (diagonal)

    Assumptions:
    - Flat-top hexes (standard for axial systems).
    - Neighbors computed with cube-coordinate equivalence for accuracy.
    - Integrates with Cell positions, Defect continuous positions (via conversion),
      and fields like BioelectricField.

    Biological context: Represents tissue lattice where cells occupy hex sites,
    enabling efficient packing and neighbor interactions mimicking epithelial sheets.
    """
    # Standard neighbor offsets for flat-top hex grid in axial coordinates
    NEIGHBOR_OFFSETS = [
        (1, 0),  # East
        (1, -1),  # Northeast
        (0, -1),  # Northwest
        (-1, 0),  # West
        (-1, 1),  # Southwest
        (0, 1)  # Southeast
    ]

    def __init__(self, width: int, height: int):
        """
        Initialize the hexagonal grid.

        Args:
            width: Number of columns (q range: 0 to width-1).
            height: Number of rows (r range: 0 to height-1).

        Generates positions as all (q, r) in the range, forming a parallelogram.
        For planarian-like shapes, external logic can mask invalid positions.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers")

        self.width = width
        self.height = height

        # List of all valid positions
        self.positions: List[Tuple[int, int]] = [
            (q, r) for q in range(width) for r in range(height)
        ]

        # Set for fast lookup
        self.position_set: Set[Tuple[int, int]] = set(self.positions)

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if a position is within the grid bounds.
        """
        q, r = pos
        return 0 <= q < self.width and 0 <= r < self.height

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get list of valid neighboring positions for a given pos.

        Uses standard axial offsets; filters for bounds.

        Returns:
            List of (q, r) tuples for neighbors (up to 6).
        """
        if not self.is_valid_position(pos):
            return []

        q, r = pos
        neighbors = []
        for dq, dr in self.NEIGHBOR_OFFSETS:
            nq, nr = q + dq, r + dr
            if self.is_valid_position((nq, nr)):
                neighbors.append((nq, nr))
        return neighbors

    def hex_to_cart(self, pos: Tuple[int, int]) -> np.ndarray:
        """
        Convert axial (q, r) to Cartesian (x, y) coordinates.

        Formula for flat-top hexes with unit spacing:
        x = q + r / 2
        y = (√3 / 2) * r

        Returns:
            np.ndarray: [x, y] float array.
        """
        q, r = pos
        x = float(q) + float(r) / 2.0
        y = (np.sqrt(3) / 2.0) * float(r)
        return np.array([x, y])

    def cart_to_hex(self, cart: np.ndarray) -> Tuple[int, int]:
        """
        Convert Cartesian (x, y) to nearest axial (q, r).

        Inverse formula (approximate; rounds to nearest).

        Args:
            cart: [x, y] array.

        Returns:
            (q, r) tuple, rounded to integers.
        """
        x, y = cart
        q = x - (y / np.sqrt(3))
        r = (2 / np.sqrt(3)) * y
        # Round to nearest integers (simple; for precision, use cube rounding)
        q_rounded = round(q)
        r_rounded = round(r)
        # Adjust if not on grid (placeholder; full cube rounding omitted for brevity)
        return (q_rounded, r_rounded)

    def distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> floating[Any]:
        """
        Compute Euclidean distance between two hex positions using Cartesian conversion.
        """
        cart1 = self.hex_to_cart(pos1)
        cart2 = self.hex_to_cart(pos2)
        return np.linalg.norm(cart1 - cart2)

    def hex_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Compute hex distance (number of steps) using cube coordinates.
        Cube: q + r + s = 0, where s = -q - r
        Distance = max(|q1 - q2|, |r1 - r2|, |s1 - s2|)
        """
        q1, r1 = pos1
        s1 = -q1 - r1
        q2, r2 = pos2
        s2 = -q2 - r2
        return max(abs(q1 - q2), abs(r1 - r2), abs(s1 - s2))

    def closest_position(self, point: np.ndarray) -> Tuple[int, int]:
        """
        Find the grid position closest to a continuous 2D point.

        Useful for mapping continuous defect positions to grid for interactions.

        Args:
            point: [x, y] continuous position.

        Returns:
            Nearest (q, r) on grid.
        """
        candidate = self.cart_to_hex(point)
        if self.is_valid_position(candidate):
            return candidate

        # If out of bounds, find minimal distance
        min_dist = np.inf
        closest = (0, 0)
        for pos in self.positions:
            dist = np.linalg.norm(self.hex_to_cart(pos) - point)
            if dist < min_dist:
                min_dist = dist
                closest = pos
        return closest

    def path_between(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Compute a shortest path between two positions using hex distance.
        Simplified linear interpolation in cube space.

        Returns:
            List of positions from start to end (inclusive).
        """
        if not (self.is_valid_position(start) and self.is_valid_position(end)):
            return []

        path = []
        n = self.hex_distance(start, end)
        if n == 0:
            return [start]

        # Cube coordinates
        q1, r1 = start
        s1 = -q1 - r1
        q2, r2 = end
        s2 = -q2 - r2

        for i in range(n + 1):
            t = i / n
            q = round(q1 * (1 - t) + q2 * t)
            r = round(r1 * (1 - t) + r2 * t)
            s = round(s1 * (1 - t) + s2 * t)
            # Adjust to satisfy q + r + s = 0 (if rounding error)
            if q + r + s != 0:
                diffs = [abs(q - (q1 * (1 - t) + q2 * t)),
                         abs(r - (r1 * (1 - t) + r2 * t)),
                         abs(s - (s1 * (1 - t) + s2 * t))]
                max_idx = np.argmax(diffs)
                if max_idx == 0:
                    q = -r - s
                elif max_idx == 1:
                    r = -q - s
                else:
                    s = -q - r
            path.append((q, r))  # s not needed
        return path