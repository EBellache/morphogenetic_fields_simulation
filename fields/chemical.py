import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter

from dynamics.cell import Cell, CellType


class MorphogenField:
    """
    Manages chemical morphogen gradient fields using reaction-diffusion dynamics.

    Implements source-sink architecture with diffusion, decay, and cell-based production.
    Morphogens are produced at sources (e.g., specific cell types or wounds) and diffuse
    across the grid, forming gradients that guide cellular behaviors like migration and differentiation.

    Attributes:
        grid_shape (Tuple[int, int]): Shape of the simulation grid (rows, columns).
        morphogens (List[str]): List of morphogen names (e.g., ['Wnt', 'BMP']).
        fields (Dict[str, np.ndarray]): Current concentration fields for each morphogen.
        diffusion_rates (Dict[str, float]): Diffusion coefficients D for each morphogen.
        decay_rates (Dict[str, float]): Decay rates λ for each morphogen.
        sources (Dict[str, Dict[Tuple[int, int], float]]): Fixed source strengths by position.
    """

    def __init__(self, grid_shape: Tuple[int, int], morphogens: List[str]):
        """
        Initializes the morphogen fields with given grid and morphogen types.

        Args:
            grid_shape (Tuple[int, int]): Grid dimensions (rows, columns).
            morphogens (List[str]): Names of morphogens to simulate.

        Raises:
            ValueError: If morphogens list is empty or grid_shape invalid.
        """
        if not morphogens:
            raise ValueError("At least one morphogen must be specified")
        if any(d <= 0 for d in grid_shape):
            raise ValueError("Grid dimensions must be positive")

        self.grid_shape = grid_shape
        self.morphogens = morphogens
        self.fields = {m: np.zeros(grid_shape, dtype=float) for m in morphogens}
        self.diffusion_rates = {m: 0.1 for m in morphogens}  # μm²/min; adjustable
        self.decay_rates = {m: 0.01 for m in morphogens}  # 1/min; adjustable
        self.sources = {m: {} for m in morphogens}  # Position to strength

    def add_source(self, morphogen: str, position: Tuple[int, int], strength: float) -> None:
        """
        Adds or updates a fixed morphogen source at a specific position.

        Sources provide constant production independent of cells (e.g., for wounds).

        Args:
            morphogen (str): Name of the morphogen.
            position (Tuple[int, int]): Grid position (row, col).
            strength (float): Source production rate (concentration units per min).

        Raises:
            ValueError: If morphogen not recognized or position out of bounds.
        """
        if morphogen not in self.morphogens:
            raise ValueError(f"Unknown morphogen: {morphogen}")
        if not (0 <= position[0] < self.grid_shape[0] and 0 <= position[1] < self.grid_shape[1]):
            raise ValueError("Position out of grid bounds")
        self.sources[morphogen][position] = strength

    def update(self, dt: float, cells: Dict[Tuple[int, int], Cell]) -> None:
        """
        Updates all morphogen fields using approximated reaction-diffusion equations.

        Solves ∂C/∂t ≈ D ∇²C - λ C + S + P, where S is fixed sources and P is cell production.
        Uses Gaussian filtering for diffusion approximation and explicit Euler for decay/production.

        Args:
            dt (float): Time step in minutes.
            cells (Dict[Tuple[int, int], Cell]): Cells by position for production computation.

        Raises:
            ValueError: If dt <= 0.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        for morphogen in self.morphogens:
            field = self.fields[morphogen].copy()
            D = self.diffusion_rates[morphogen]
            λ = self.decay_rates[morphogen]

            # Approximate diffusion with Gaussian filter (sigma = sqrt(2 D dt))
            if D > 0:
                sigma = np.sqrt(2 * D * dt)
                diffused = gaussian_filter(field, sigma=sigma, mode='constant')
            else:
                diffused = field

            # Apply decay: C * exp(-λ dt) ≈ C * (1 - λ dt) for small dt, but use exact
            decayed = diffused * np.exp(-λ * dt)

            # Add fixed sources: S * dt (assuming constant over dt)
            production = np.zeros(self.grid_shape, dtype=float)
            for pos, strength in self.sources[morphogen].items():
                production[pos] += strength * dt

            # Add cell-based production
            for pos, cell in cells.items():
                if self._cell_produces(cell, morphogen):
                    rate = self._production_rate(cell, morphogen)
                    # Handle coordinate conversion and bounds checking
                    if isinstance(pos, tuple) and len(pos) == 2:
                        y, x = pos if pos[0] < self.grid_shape[0] else (pos[1], pos[0])
                        if 0 <= y < self.grid_shape[0] and 0 <= x < self.grid_shape[1]:
                            production[y, x] += rate * dt

            # Update field
            self.fields[morphogen] = decayed + production

    def _cell_produces(self, cell: Cell, morphogen: str) -> bool:
        """
        Checks if a cell produces a given morphogen based on type.

        Production map defines which types act as sources for each morphogen.

        Args:
            cell (Cell): The cell to check.
            morphogen (str): Morphogen name.

        Returns:
            bool: True if cell produces the morphogen.
        """
        production_map = {
            'Wnt': [CellType.STEM.name, CellType.EPITHELIAL.name],
            'BMP': [CellType.EPITHELIAL.name, CellType.CONNECTIVE.name],
            'Shh': [CellType.NEURAL.name],
            'FGF': [CellType.MUSCLE.name, CellType.CONNECTIVE.name]
        }
        return cell.cell_type.name in production_map.get(morphogen, [])

    def _production_rate(self, cell: Cell, morphogen: str) -> float:
        """
        Computes the production rate of a morphogen by a cell.

        Base rates modulated by cell state (e.g., stem cells produce more) and wound signals.

        Args:
            cell (Cell): Producing cell.
            morphogen (str): Morphogen name.

        Returns:
            float: Production rate (concentration units per min).
        """
        base_rates = {
            'Wnt': 1.0,
            'BMP': 0.5,
            'Shh': 0.8,
            'FGF': 0.6
        }
        rate = base_rates.get(morphogen, 0.1)
        if cell.cell_type == CellType.STEM:
            rate *= 1.5  # Stem cells as strong sources
        rate *= (1 + cell.wound_signal_strength)  # Wound amplification
        return rate

    def get_gradient(self, morphogen: str, position: Tuple[int, int]) -> np.ndarray:
        """
        Computes the local gradient of a morphogen field at a position using finite differences.

        Handles boundary cases with one-sided differences.

        Args:
            morphogen (str): Morphogen name.
            position (Tuple[int, int]): Grid position (row, col).

        Returns:
            np.ndarray: 2D gradient vector [dx, dy].

        Raises:
            ValueError: If morphogen unknown or position invalid.
        """
        if morphogen not in self.fields:
            raise ValueError(f"Unknown morphogen: {morphogen}")
        if not (0 <= position[0] < self.grid_shape[0] and 0 <= position[1] < self.grid_shape[1]):
            raise ValueError("Position out of grid bounds")

        field = self.fields[morphogen]
        y, x = position

        # dx: central difference if possible, else one-sided
        if x > 0 and x < field.shape[1] - 1:
            dx = (field[y, x + 1] - field[y, x - 1]) / 2.0
        elif x > 0:
            dx = field[y, x] - field[y, x - 1]
        elif x < field.shape[1] - 1:
            dx = field[y, x + 1] - field[y, x]
        else:
            dx = 0.0

        # dy: similar for rows
        if y > 0 and y < field.shape[0] - 1:
            dy = (field[y + 1, x] - field[y - 1, x]) / 2.0
        elif y > 0:
            dy = field[y, x] - field[y - 1, x]
        elif y < field.shape[0] - 1:
            dy = field[y + 1, x] - field[y, x]
        else:
            dy = 0.0

        return np.array([dx, dy])