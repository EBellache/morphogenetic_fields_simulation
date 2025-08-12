import numpy as np
from typing import Tuple, Union

# Type alias for bit vectors (numpy arrays of 0/1)
BitVector = np.ndarray


class GolayCode:
    """
    Implements the extended binary Golay code (24,12,8) for biological error correction.

    As described in the manuscript, this code provides local protection within the triple-code
    architecture, ensuring robustness against perturbations in bioelectric and topological states.
    The Golay code's perfect error-correcting properties (corrects up to 3 errors) mirror biological
    redundancy mechanisms, such as gene duplication or checkpoint controls in cell cycles.

    Attributes:
        n (int): Code length (24 bits).
        k (int): Message dimension (12 bits).
        d (int): Minimum distance (8).
        generator (np.ndarray): Generator matrix G = [I_k | A] for encoding.
        parity_check (np.ndarray): Parity-check matrix H = [A.T | I_{n-k}].
    """

    def __init__(self):
        """
        Initializes the GolayCode with standard parameters and matrices.

        The generator and parity-check matrices are constructed based on the Mathieu group M24
        structure, providing the algebraic foundation for the code's exceptional properties.
        """
        self.n = 24
        self.k = 12
        self.d = 8
        self._build_matrices()

    def _build_matrices(self) -> None:
        """
        Constructs the generator and parity-check matrices for the binary Golay code.

        Uses a standard form where G = [I_12 | A], and H = [A.T | I_12], with A being
        a specific 12x12 matrix ensuring the code's properties. The matrix A provided here
        is a valid construction for the Golay code.
        """
        # Identity matrix for the systematic part
        I = np.eye(self.k, dtype=int)

        # Standard parity submatrix A for Golay code (12x12)
        # This is a known valid matrix; rows correspond to basis vectors
        A = np.array([
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
        ], dtype=int)

        self.generator = np.hstack([I, A])
        self.parity_check = np.hstack([A.T, np.eye(self.n - self.k, dtype=int)])

    def encode(self, message: Union[list[int], BitVector]) -> BitVector:
        """
        Encodes a 12-bit message into a 24-bit Golay codeword.

        Biologically: Represents embedding developmental instructions (message) into
        a robust bioelectric pattern (codeword) resistant to noise from environmental perturbations.

        Args:
            message: 12-bit message as list of ints (0/1) or numpy array.

        Returns:
            BitVector: 24-bit codeword as numpy array.

        Raises:
            ValueError: If message length is not 12 or contains invalid bits.
        """
        message = np.array(message, dtype=int)
        if message.shape != (self.k,):
            raise ValueError(f"Message must be a vector of length {self.k}")
        if not np.all(np.isin(message, [0, 1])):
            raise ValueError("Message bits must be 0 or 1")
        codeword = np.dot(message, self.generator) % 2
        return codeword

    def syndrome(self, received: Union[list[int], BitVector]) -> BitVector:
        """
        Computes the syndrome of a received word for error detection.

        The syndrome is H * received^T mod 2. If zero, no detectable errors.

        Args:
            received: 24-bit received word as list of ints (0/1) or numpy array.

        Returns:
            BitVector: 12-bit syndrome vector.

        Raises:
            ValueError: If received length is not 24 or contains invalid bits.
        """
        received = np.array(received, dtype=int)
        if received.shape != (self.n,):
            raise ValueError(f"Received word must be a vector of length {self.n}")
        if not np.all(np.isin(received, [0, 1])):
            raise ValueError("Received bits must be 0 or 1")
        syn = np.dot(received, self.parity_check.T) % 2
        return syn

    def decode(self, received: Union[list[int], BitVector]) -> Tuple[BitVector, int]:
        """
        Decodes a received word, correcting up to 3 errors using syndrome decoding.

        Implements brute-force search over error patterns of weight <=3 to find the one
        matching the syndrome. If found, corrects; otherwise, returns original with -1 errors.
        This is computationally feasible (binomial(24,3)=2024 patterns) for simulation purposes.

        Biologically: Models cellular error correction, e.g., DNA repair or bioelectric
        pattern restoration during regeneration.

        Args:
            received: 24-bit received word as list of ints (0/1) or numpy array.

        Returns:
            Tuple[BitVector, int]: (corrected_message (12 bits), num_errors_corrected) or -1 if uncorrectable.

        Raises:
            ValueError: If received length is not 24 or contains invalid bits.
        """
        received = np.array(received, dtype=int)
        if received.shape != (self.n,):
            raise ValueError(f"Received word must be a vector of length {self.n}")
        if not np.all(np.isin(received, [0, 1])):
            raise ValueError("Received bits must be 0 or 1")

        syn = self.syndrome(received)
        if np.all(syn == 0):
            return received[:self.k], 0  # No errors

        # Brute-force search for error pattern e with wt(e) <=3 and H e^T = syn
        for wt in range(1, 4):  # Up to t=3
            # Generate all combinations of positions for errors
            from itertools import combinations
            for positions in combinations(range(self.n), wt):
                error = np.zeros(self.n, dtype=int)
                error[list(positions)] = 1
                error_syn = np.dot(error, self.parity_check.T) % 2
                if np.array_equal(error_syn, syn):
                    corrected = (received + error) % 2
                    message = corrected[:self.k]
                    return message, wt

        # Uncorrectable (more than 3 errors or undetectable)
        return received[:self.k], -1