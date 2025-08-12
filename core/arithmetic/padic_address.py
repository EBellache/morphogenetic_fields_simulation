import numpy as np
from typing import List, Tuple, Optional


class PadicAddress:
    """
    Represents a p-adic address encoding a cell's developmental history.

    The address is stored as a list of digits [d0, d1, d2, ...] where d0 is the least significant digit (earliest branching),
    each di ∈ {0, 1, ..., p-1}, and p is prime.

    Attributes:
        prime (int): The base prime p (must be 2,3,5, or 7).
        digits (List[int]): List of digits, least significant first.
        level (int): Developmental level (length of digits list).
    """
    ALLOWED_PRIMES = {2, 3, 5, 7}

    def __init__(self, prime: int, digits: Optional[List[int]] = None):
        """
        Initialize a p-adic address.

        Args:
            prime: The base prime p.
            digits: Initial list of digits (least significant first). Defaults to empty (root address).

        Raises:
            ValueError: If prime not allowed or digits invalid.
        """
        if prime not in self.ALLOWED_PRIMES:
            raise ValueError(f"Prime must be one of {self.ALLOWED_PRIMES}")
        self.prime = prime

        if digits is None:
            digits = []
        self.digits = digits.copy()
        self._validate_digits()

        self.level = len(self.digits)  # Developmental depth/hierarchy scale

    def _validate_digits(self) -> None:
        """Ensure all digits are in {0, 1, ..., p-1}."""
        for d in self.digits:
            if not (0 <= d < self.prime):
                raise ValueError(f"Digit {d} not in range [0, {self.prime - 1}]")

    def valuation(self) -> int:
        """
        Compute the p-adic valuation: number of trailing zeros (from LSB).
        Biologically: measures "stemness" or undifferentiated potential.
        """
        val = 0
        for d in self.digits:
            if d == 0:
                val += 1
            else:
                break
        return val

    def branch(self, direction: int) -> 'PadicAddress':
        """
        Create a child address by appending a new digit (branching decision).

        Args:
            direction: Digit to append (0 to p-1), representing branching choice.
            Prime-specific:
                - p=2: 0=left/posterior, 1=right/anterior
                - p=3: 0=ecto, 1=meso, 2=endo
                - p=5: Directions follow golden ratio angles
                - p=7: Exceptional symmetries (e.g., E8 root choices)

        Returns:
            New PadicAddress with appended digit.

        Raises:
            ValueError: If direction invalid.
        """
        if not (0 <= direction < self.prime):
            raise ValueError(f"Direction must be in [0, {self.prime - 1}]")
        new_digits = self.digits + [direction]
        return PadicAddress(self.prime, new_digits)

    def distance(self, other: 'PadicAddress') -> float:
        """
        Compute p-adic distance (ultrametric) between two addresses.
        d(x,y) = p^{-v_p(x-y)}, where v_p is the valuation of difference.
        If addresses have different lengths, pad shorter with zeros (extend to common length).

        Biologically: Measures developmental divergence; ultrametric implies hierarchical clustering.

        Args:
            other: Another PadicAddress with same prime.

        Returns:
            float: p-adic distance (0 if identical, small if recent divergence).

        Raises:
            ValueError: If primes differ.
        """
        if self.prime != other.prime:
            raise ValueError("Addresses must have same prime")

        # Pad shorter list with zeros (extend to infinite series with trailing zeros)
        len1, len2 = len(self.digits), len(other.digits)
        max_len = max(len1, len2)
        d1 = self.digits + [0] * (max_len - len1)
        d2 = other.digits + [0] * (max_len - len2)

        # Find minimal valuation of difference
        v = 0
        for a, b in zip(d1, d2):
            if a != b:
                break
            v += 1
        return self.prime ** (-v) if v < max_len else 0.0  # 0 if identical up to max_len

    def common_ancestor(self, other: 'PadicAddress') -> Tuple[int, 'PadicAddress']:
        """
        Find the common ancestor: branching point where addresses diverge.

        Args:
            other: Another PadicAddress with same prime.

        Returns:
            Tuple: (divergence_level, ancestor_address)
            divergence_level: Index where paths split (-1 if identical).

        Raises:
            ValueError: If primes differ.
        """
        if self.prime != other.prime:
            raise ValueError("Addresses must have same prime")

        min_len = min(len(self.digits), len(other.digits))
        for i in range(min_len):
            if self.digits[i] != other.digits[i]:
                ancestor_digits = self.digits[:i]
                return i, PadicAddress(self.prime, ancestor_digits)

        # If one is prefix of the other, divergence at min_len
        return min_len, PadicAddress(self.prime, self.digits[:min_len])

    def cell_type_modular(self, modulus: int) -> int:
        """
        Determine cell type via modular arithmetic on digits.
        Example: sum(digits) mod modulus for discrete types.

        Biologically: Maps address to fate (e.g., modulus=3 for germ layers).

        Args:
            modulus: Modular base for type determination.

        Returns:
            int: Cell type index (0 to modulus-1).
        """
        if not self.digits:
            return 0  # Root cell type
        total = sum(self.digits)
        return total % modulus

    def is_self_similar(self, period: int) -> bool:
        """
        Check for self-similar (repeating) pattern with given period.
        Biologically: Indicates stable cell type across scales.

        Args:
            period: Repeat length to check.

        Returns:
            bool: True if digits repeat every 'period' steps.
        """
        if period == 0 or len(self.digits) < 2 * period:
            return False
        pattern = self.digits[:period]
        for i in range(1, len(self.digits) // period):
            if self.digits[i * period:(i + 1) * period] != pattern:
                return False
        return True

    def log_periodic_modulation(self, scale: float) -> float:
        """
        Compute log-periodic modulation based on level.
        Biologically: Introduces oscillatory behavior in fractal growth (e.g., segmentation).

        Args:
            scale: Scaling factor for periodicity.

        Returns:
            float: Modulation value in [-1,1].
        """
        if self.level == 0:
            return 0.0
        log_level = np.log(self.level) / np.log(self.prime)
        return np.sin(2 * np.pi * log_level / scale)

    def __repr__(self) -> str:
        """String representation: p-adic address as ...d2 d1 d0 (LSB right)."""
        if not self.digits:
            return f"PadicAddress(p={self.prime}, root)"
        return f"PadicAddress(p={self.prime}, digits={self.digits[::-1]})"  # MSB first for readability

    def __eq__(self, other: object) -> bool:
        """Equality: same prime and digits (padded with zeros if needed)."""
        if not isinstance(other, PadicAddress):
            return False
        if self.prime != other.prime:
            return False
        # Compare with padding
        len_diff = len(self.digits) - len(other.digits)
        if len_diff > 0:
            return self.digits[:len(other.digits)] == other.digits and all(
                d == 0 for d in self.digits[len(other.digits):])
        elif len_diff < 0:
            return other.digits[:len(self.digits)] == self.digits and all(
                d == 0 for d in other.digits[len(self.digits):])
        else:
            return self.digits == other.digits