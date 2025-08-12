from enum import Enum
from typing import List
import numpy as np

class TopologicalInvariant(Enum):
    """
    Enumeration of topological invariants used to control regeneration processes.
    These invariants ensure structural integrity and guide halting conditions
    during morphogenetic simulations.
    """
    EULER_CHARACTERISTIC = "euler"
    WINDING_NUMBER = "winding"
    LINKING_NUMBER = "linking"
    HOPF_INVARIANT = "hopf"

class IndexTheorem:
    """
    Implements index theorems for determining regeneration stopping conditions
    (as described in Section 5.3 of the manuscript). The system halts when the
    computed global index matches the target topology, ensuring convergence to
    the desired morphological state.

    Attributes:
        genus (int): Genus of the surface (default: 0 for sphere-like topology).
        boundaries (int): Number of boundary components (default: 0 for closed surfaces).
        euler_char (int): Computed Euler characteristic χ = 2 - 2g - b.
    """
    def __init__(self, surface_genus: int = 0, boundary_components: int = 0):
        """
        Initializes the IndexTheorem with surface topology parameters.

        Args:
            surface_genus (int, optional): Genus g of the surface. Defaults to 0.
            boundary_components (int, optional): Number of boundaries b. Defaults to 0.
        """
        if surface_genus < 0 or boundary_components < 0:
            raise ValueError("Genus and boundary components must be non-negative integers")
        self.genus = surface_genus
        self.boundaries = boundary_components
        self.euler_char = 2 - 2 * self.genus - self.boundaries

    def poincare_hopf_index(self, defect_charges: List[float]) -> float:
        """
        Computes the Poincaré-Hopf index for a collection of defect charges.

        The index is given by Σ q_k - χ + (boundary curvature integral) / (2π),
        where χ is the Euler characteristic. For closed surfaces, this simplifies
        to Σ q_k - χ, and regeneration stops when this value approaches zero,
        indicating topological equilibrium.

        Args:
            defect_charges (List[float]): List of quantized defect charges (e.g., from disclinations).

        Returns:
            float: Computed Poincaré-Hopf index.
        """
        total_charge = sum(defect_charges)
        boundary_term = self._compute_boundary_curvature()
        return total_charge - self.euler_char + boundary_term / (2 * np.pi)

    def _compute_boundary_curvature(self) -> float:
        """
        Computes the integral of geodesic curvature over all boundary components.

        This is a simplified model assuming each boundary is a unit circle, yielding
        a curvature integral of 2π per boundary. For more complex geometries, this
        method could be extended with explicit boundary data.

        Returns:
            float: Total boundary curvature integral.
        """
        return 2 * np.pi * self.boundaries

    def check_stopping_condition(self, defect_charges: List[float], tolerance: float = 1e-6) -> bool:
        """
        Checks if the regeneration process should halt based on the Poincaré-Hopf index.

        Halting occurs when the absolute value of the index falls below a specified
        numerical tolerance, signifying that the defect configuration matches the
        target topology.

        Args:
            defect_charges (List[float]): List of defect charges.
            tolerance (float, optional): Numerical tolerance for index comparison. Defaults to 1e-6.

        Returns:
            bool: True if the stopping condition is met, False otherwise.
        """
        index = self.poincare_hopf_index(defect_charges)
        return abs(index) < tolerance