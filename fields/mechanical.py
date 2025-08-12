import numpy as np
from typing import Dict, Tuple, Optional
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from core.geometry.hexagonal_grid import HexagonalGrid
from dynamics.cell import Cell


class MechanicalField:
    """
    Simulates mechanical stress and strain fields in tissue using constructal flow constraints.

    As described in the manuscript, this class models tissue mechanics via elasticity equations,
    solving for pressure distributions and computing stress tensors. It integrates with hexagonal
    grids for biological accuracy, incorporating cell-generated forces and boundary conditions.

    Attributes:
        grid (HexagonalGrid): The underlying hexagonal grid.
        pressure (np.ndarray): Scalar pressure field (shape: grid positions flattened or 2D mapped).
        stress_tensor (np.ndarray): 2x2 stress tensors per position (shape: (n_points, 2, 2)).
        strain_rate (np.ndarray): 2x2 strain rate tensors per position (shape: (n_points, 2, 2)).
        young_modulus (float): Young's modulus in kPa.
        poisson_ratio (float): Poisson's ratio.
        viscosity (float): Viscosity in Pa·s.
    """

    def __init__(self, grid: HexagonalGrid, young_modulus: float = 1.0, poisson_ratio: float = 0.3,
                 viscosity: float = 1.0):
        """
        Initializes the mechanical field on a hexagonal grid with material properties.

        Args:
            grid (HexagonalGrid): The hexagonal grid defining positions.
            young_modulus (float, optional): Elastic modulus in kPa. Defaults to 1.0.
            poisson_ratio (float, optional): Poisson's ratio. Defaults to 0.3.
            viscosity (float, optional): Viscosity in Pa·s. Defaults to 1.0.
        """
        self.grid = grid
        self.n_points = len(grid.positions)
        self.pos_to_idx = {pos: i for i, pos in enumerate(grid.positions)}
        self.idx_to_pos = {i: pos for pos, i in self.pos_to_idx.items()}

        self.pressure = np.zeros(self.n_points)
        self.stress_tensor = np.zeros((self.n_points, 2, 2))
        self.strain_rate = np.zeros((self.n_points, 2, 2))

        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.viscosity = viscosity

    def update(self, dt: float, cells: Dict[Tuple[int, int], Cell],
               boundaries: Optional[Dict[Tuple[int, int], float]] = None) -> None:
        """
        Updates mechanical fields by solving the pressure Poisson equation and computing stress.

        Equation: ∇²P = -div(F), where F includes cell growth forces. Uses sparse linear algebra
        for efficiency on hexagonal topology. Boundaries can fix pressures (Dirichlet conditions).

        Args:
            dt (float): Time step (unused in static solve; for future viscoelastic extensions).
            cells (Dict[Tuple[int, int], Cell]): Cells by position for force sources.
            boundaries (Optional[Dict[Tuple[int, int], float]]): Fixed pressures by position.

        Raises:
            ValueError: If dt <= 0 (for consistency, even if unused).
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        A = lil_matrix((self.n_points, self.n_points))
        b = np.zeros(self.n_points)

        # Build discrete Laplacian on hex grid (stencil: -deg on diagonal, 1 on neighbors)
        for idx in range(self.n_points):
            pos = self.idx_to_pos[idx]
            neighbors = self.grid.get_neighbors(pos)
            deg = len(neighbors)
            A[idx, idx] = -deg  # Central coefficient

            for neigh_pos in neighbors:
                neigh_idx = self.pos_to_idx.get(neigh_pos)
                if neigh_idx is not None:
                    A[idx, neigh_idx] = 1.0

            # Source term: negative divergence of forces (from cell division accumulators)
            if pos in cells:
                cell = cells[pos]
                b[idx] = -cell.division_accumulator  # Growth pressure as source

        # Apply Dirichlet boundary conditions if provided
        if boundaries:
            for pos, fixed_p in boundaries.items():
                if pos in self.pos_to_idx:
                    idx = self.pos_to_idx[pos]
                    A[idx, :] = 0
                    A[idx, idx] = 1.0
                    b[idx] = fixed_p

        # Solve A p = b for pressure
        A_csc = A.tocsc()
        self.pressure = spsolve(A_csc, b)

        # Update stress tensor
        self._compute_stress(cells)

    def _compute_stress(self, cells: Dict[Tuple[int, int], Cell]) -> None:
        """
        Computes the stress tensor σ = -P I + 2μ ε, where ε is strain rate (placeholder).

        Incorporates isotropic pressure and deviatoric terms from cell adhesion.

        Args:
            cells (Dict[Tuple[int, int], Cell]): Cells for position-specific properties.
        """
        for idx in range(self.n_points):
            pos = self.idx_to_pos[idx]
            P = self.pressure[idx]

            # Isotropic pressure contribution
            stress = -P * np.eye(2)

            # Add deviatoric stress from cell properties if present
            if pos in cells:
                cell = cells[pos]
                adhesion = cell.mechanical_state.adhesion
                # Assume adhesion adds equal tension in x and y (isotropic)
                stress[0, 0] += adhesion
                stress[1, 1] += adhesion

            # Placeholder for viscous term: 2 η ε (strain rate); here ε=0
            self.stress_tensor[idx] = stress

    def compute_traction(self, position: Tuple[int, int]) -> np.ndarray:
        """
        Computes the traction force vector at a position as div(σ).

        Uses finite differences on hex grid neighbors for divergence approximation.

        Args:
            position (Tuple[int, int]): Grid position.

        Returns:
            np.ndarray: 2D traction vector.

        Raises:
            ValueError: If position invalid.
        """
        if position not in self.pos_to_idx:
            raise ValueError("Invalid position")
        idx = self.pos_to_idx[position]
        neighbors = self.grid.get_neighbors(position)
        if not neighbors:
            return np.zeros(2)

        traction = np.zeros(2)
        for neigh_pos in neighbors:
            if neigh_pos in self.pos_to_idx:
                neigh_idx = self.pos_to_idx[neigh_pos]
                # Direction vector (unit)
                dir_vec = np.array(neigh_pos) - np.array(position)
                dist = np.linalg.norm(dir_vec) + 1e-6
                unit_dir = dir_vec / dist
                # Stress difference projection (simplified div(σ) ≈ sum (σ_neigh - σ) · unit_dir / deg)
                stress_diff = self.stress_tensor[neigh_idx] - self.stress_tensor[idx]
                traction += np.dot(stress_diff, unit_dir)

        traction /= len(neighbors)  # Average over neighbors
        return traction