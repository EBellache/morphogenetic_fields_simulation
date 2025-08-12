import numpy as np
from typing import List, Tuple


class FractalBranching:
    """
    Implements fractal cell fate determination rules using p-adic structures.

    As described in Section 7.1 of the manuscript, this class models the recursive
    blow-up operator, which transforms topological singularities into complex
    hierarchical structures. This process represents the causal tree of development,
    where branching decisions at each level are governed by prime numbers, reflecting
    biological symmetries such as bilateral (p=2) or tripartite (p=3) patterning.

    Attributes:
        primes (List[int]): List of primes used for branching factors at each level
            (default: [2, 3, 5, 7], corresponding to common biological multiplicities).
    """

    def __init__(self, primes: List[int] = [2, 3, 5, 7]):
        """
        Initializes the FractalBranching with a list of primes for hierarchical levels.

        Args:
            primes (List[int], optional): Primes determining branching arity at each level.
                Defaults to [2, 3, 5, 7].

        Raises:
            ValueError: If primes list is empty or contains non-prime/invalid values.
        """
        if not primes:
            raise ValueError("Primes list must be non-empty")
        for p in primes:
            if not self._is_prime(p) or p < 2:
                raise ValueError(f"Invalid prime: {p}. Must be prime numbers >= 2")
        self.primes = primes
        # Cache for blow-up results to optimize recursive calls (memoization)
        self.blow_up_cache: dict[Tuple[Tuple[int, int], int], List[Tuple[int, int]]] = {}

    @staticmethod
    def _is_prime(n: int) -> bool:
        """Helper to check if a number is prime."""
        if n <= 1:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def blow_up_operator(self, singularity_pos: Tuple[int, int], level: int) -> List[Tuple[int, int]]:
        """
        Applies the recursive blow-up operator to a singularity position.

        This operator generates a fractal tree of positions, simulating the emergence
        of complex structures from defects during morphogenesis. At each level, the
        branching factor is determined by the corresponding prime, with offsets
        distributed angularly to mimic symmetric growth patterns.

        Uses memoization to cache results for efficiency in deep recursions.

        Args:
            singularity_pos (Tuple[int, int]): Starting position (x, y) of the singularity.
            level (int): Recursion depth (number of branching levels).

        Returns:
            List[Tuple[int, int]]: List of all generated positions in the fractal tree.

        Raises:
            ValueError: If level is negative.
        """
        if level < 0:
            raise ValueError("Level must be non-negative")

        key = (singularity_pos, level)
        if key in self.blow_up_cache:
            return self.blow_up_cache[key]

        if level == 0:
            result = [singularity_pos]
        else:
            # Select prime for this level (cycle if level exceeds list length)
            prime_idx = (level - 1) % len(self.primes)
            prime = self.primes[prime_idx]

            result = []
            for i in range(prime):
                # Angular distribution for symmetric branching
                angle = 2 * np.pi * i / prime
                # Scale offset with level (exponential growth for fractal scaling)
                scale = 2 ** (level - 1)  # Power of 2 for binary-like scaling
                dx = int(np.cos(angle) * scale)
                dy = int(np.sin(angle) * scale)
                child_pos = (singularity_pos[0] + dx, singularity_pos[1] + dy)
                # Recurse to lower level
                result.extend(self.blow_up_operator(child_pos, level - 1))

        self.blow_up_cache[key] = result
        return result

    def log_periodic_modulation(self, scale: float, phase: float = 0.0) -> float:
        """
        Computes log-periodic modulation based on scale, derived from spectral zeta poles.

        As per the manuscript, this introduces measurable oscillations in morphogenetic
        processes, such as periodic gene expression waves or segmented growth patterns.
        The modulation is periodic in the logarithm of scale, reflecting fractal self-similarity.

        Args:
            scale (float): Positive scale factor (e.g., time or size in development).
            phase (float, optional): Phase shift for the oscillation. Defaults to 0.0.

        Returns:
            float: Modulation value in [-1, 1].

        Raises:
            ValueError: If scale is non-positive.
        """
        if scale <= 0:
            raise ValueError("Scale must be positive")
        log_scale = np.log(scale)
        # Use the first prime as the base period (binary oscillations common in biology)
        period = np.log(self.primes[0])
        return np.sin(2 * np.pi * log_scale / period + phase)