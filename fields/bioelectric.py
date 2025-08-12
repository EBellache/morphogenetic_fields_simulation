import numpy as np
from typing import Dict, Tuple, Optional, Any
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from core.geometry.hexagonal_grid import HexagonalGrid  # Assuming this exists for grid management
from dynamics.cell import Cell
from core.topology.defect import DefectSystem, Defect


class BioelectricField:
    """
    Implements the bioelectric field system for long-range coordination in regeneration.

    Manages voltage field V and current density J on a hexagonal lattice.
    Solves the cable equation: ∂V/∂t = D ∇²V - (V - V_rest)/τ + I_external

    Integrates with cells (voltage storage), defects (conductance modulation),
    and provides gradients for migration/division cues.
    """

    def __init__(
            self,
            grid: HexagonalGrid,
            D: float = 1.0,  # Diffusion coefficient (gap junction conductance)
            tau: float = 1.0,  # Membrane time constant
            V_rest: float = -70.0,  # Resting potential in mV
            defect_system: Optional[DefectSystem] = None
    ):
        self.grid = grid
        self.D = D
        self.tau = tau
        self.V_rest = V_rest
        self.defect_system = defect_system or DefectSystem()

        # Voltage storage: dict keyed by hex position (q,r)
        self.voltage: Dict[Tuple[int, int], float] = {}
        self.current_density: Dict[Tuple[int, int], np.ndarray] = {}  # 2D vector J

        # Fixed potential sites (pacemakers)
        self.fixed_potentials: Dict[Tuple[int, int], float] = {}

        # Wound sites: current sources
        self.wound_currents: Dict[Tuple[int, int], float] = {}

        # Initialize voltages to V_rest
        for pos in self.grid.positions:
            self.voltage[pos] = V_rest
            self.current_density[pos] = np.zeros(2)

    def set_fixed_potential(self, pos: Tuple[int, int], V_fixed: float) -> None:
        """Set a fixed potential at a position (e.g., pacemaker cell)."""
        if pos in self.grid.positions:
            self.fixed_potentials[pos] = V_fixed

    def add_wound_current(self, pos: Tuple[int, int], I_wound: float) -> None:
        """Add a current source at wound site (depolarization)."""
        if pos in self.grid.positions:
            self.wound_currents[pos] = I_wound

    def update(self, dt: float, cells: Dict[Tuple[int, int], Cell]) -> None:
        """
        Update the voltage field using implicit solver for stability.
        Then update cell voltages and compute currents.

        Args:
            dt: Timestep.
            cells: Dict of cells at positions for coupling.
        """
        N = len(self.grid.positions)
        pos_to_idx = {pos: i for i, pos in enumerate(self.grid.positions)}
        idx_to_pos = {i: pos for pos, i in pos_to_idx.items()}

        # Build sparse matrix for discrete cable equation
        A = lil_matrix((N, N))
        b = np.zeros(N)

        for i in range(N):
            pos = idx_to_pos[i]
            if pos in self.fixed_potentials:
                A[i, i] = 1.0
                b[i] = self.fixed_potentials[pos]
                continue

            # Time discretization: (V^{n+1} - V^n)/dt = D ∇²V^{n+1} - (V^{n+1} - V_rest)/τ + I_ext
            # Rearrange: (1/dt + 1/τ) V - D ∇²V = V^n / dt + V_rest / τ + I_ext

            coeff_self = 1 / dt + 1 / self.tau
            A[i, i] = coeff_self

            # Discrete Laplacian on hex grid: ∇²V ≈ (2/3h²) * Σ (V_n - V) over 6 neighbors (for hex spacing h=1)
            # But simplify to average: ∇²V ≈ (1/3) Σ (V_n - V) since D absorbs scaling
            neighbors = self.grid.get_neighbors(pos)
            num_neigh = len(neighbors)
            if num_neigh > 0:
                lap_coeff = self.D / num_neigh  # Normalized
                A[i, i] += self.D  # -D * num_neigh / num_neigh for central
                for neigh_pos in neighbors:
                    if neigh_pos in pos_to_idx:
                        j = pos_to_idx[neigh_pos]
                        A[i, j] = -lap_coeff

            # Right-hand side
            V_current = self.voltage[pos]
            b[i] = V_current / dt + self.V_rest / self.tau

            # External current (wounds)
            if pos in self.wound_currents:
                b[i] += self.wound_currents[pos]

            # Defect modulation: reduce conductance near defects
            if self.defect_system:
                dist_to_defect = min(
                    [np.linalg.norm(np.array(pos) - d.position) for d in self.defect_system.defects]
                    or [np.inf]
                )
                if dist_to_defect < 2.0:  # Arbitrary threshold
                    A[i, i] *= 0.5  # Halve local conductance (alter self term)

        # Solve A V = b
        A = A.tocsc()  # Convert to CSC for solver
        V_new = spsolve(A, b)

        # Update voltages
        for i in range(N):
            pos = idx_to_pos[i]
            self.voltage[pos] = V_new[i]

            # Sync to cell if present
            if pos in cells:
                cells[pos].bioelectric_state.voltage = V_new[i]

        # Compute current density J = -D ∇V (simplified Ohmic)
        self._compute_currents()

        # Defect drift: move defects along voltage gradients
        self._drift_defects(dt)

        # Check for topological transitions at critical potentials
        self._check_transitions(cells)

    def _compute_currents(self) -> None:
        """Compute current density J at each point based on local ∇V."""
        for pos in self.grid.positions:
            grad_V = self.get_gradient_at(pos)
            self.current_density[pos] = -self.D * grad_V  # J = -σ ∇V, with σ=D here

    def get_gradient_at(self, pos: Tuple[int, int]) -> np.ndarray:
        """
        Compute voltage gradient at position using finite differences on hex grid.

        Returns:
            np.ndarray: 2D gradient vector.
        """
        neighbors = self.grid.get_neighbors(pos)
        if not neighbors:
            return np.zeros(2)

        # Convert hex positions to Cartesian for gradient estimation
        # Assuming standard hex to cart: x = q + r/2, y = (√3/2) r
        def hex_to_cart(p: Tuple[int, int]) -> np.ndarray:
            q, r = p
            return np.array([q + r / 2, (np.sqrt(3) / 2) * r])

        pos_cart = hex_to_cart(pos)
        grad = np.zeros(2)
        for neigh in neighbors:
            if neigh in self.voltage:
                neigh_cart = hex_to_cart(neigh)
                delta_r = neigh_cart - pos_cart
                delta_V = self.voltage[neigh] - self.voltage[pos]
                unit_r = delta_r / (np.linalg.norm(delta_r) + 1e-6)
                grad += delta_V * unit_r  # Weighted by direction
        return grad / len(neighbors) if neighbors else np.zeros(2)

    def _drift_defects(self, dt: float) -> None:
        """Drift defects along voltage gradients (defects move toward depolarization)."""
        for defect in self.defect_system.defects:
            # Find nearest grid position
            nearest_pos = min(
                self.grid.positions,
                key=lambda p: np.linalg.norm(np.array(p) - defect.position)
            )
            grad_V = self.get_gradient_at(nearest_pos)
            drift_vel = 0.1 * grad_V  # Arbitrary mobility; toward positive grad (depolarization)
            defect.position += drift_vel * dt

    def _check_transitions(self, cells: Dict[Tuple[int, int], Cell]) -> None:
        """Check for topological transitions at critical potentials."""
        for pos, V in self.voltage.items():
            if V > -40.0:  # Depolarized threshold
                # Trigger creation of +1/2 disclination or similar
                if np.random.random() < 0.01:  # Probabilistic for simulation
                    self.defect_system.create_defect_pair(
                        pos1=pos, charge1=0.5, type1=DefectType.DISCLINATION,
                        pos2=(pos[0] + 1, pos[1]), charge2=-0.5, type2=DefectType.DISCLINATION
                    )
            elif V < -90.0:  # Hyperpolarized
                # Potential annihilation trigger (placeholder)
                pass

    def set_pattern(self, anterior_pos: Tuple[int, int], posterior_pos: Tuple[int, int]) -> None:
        """Set anterior-posterior gradient pattern."""
        self.set_fixed_potential(anterior_pos, -40.0)  # Depolarized
        self.set_fixed_potential(posterior_pos, -90.0)  # Hyperpolarized

    def perturb(self, pos: Tuple[int, int], delta_V: float) -> None:
        """Apply experimental perturbation to voltage at position."""
        if pos in self.voltage:
            self.voltage[pos] += delta_V

    def get_biological_effects(self, pos: Tuple[int, int]) -> Dict[str, float]:
        """
        Compute biological cues from local voltage.

        Returns:
            Dict with 'division_rate', 'migration_dir' (vector), 'diff_fate' (threshold), 'apoptosis_prob'.
        """
        if pos not in self.voltage:
            return {}

        V = self.voltage[pos]
        grad_V = self.get_gradient_at(pos)

        # Division: depolarization promotes proliferation
        division_rate = max(0.0, (V + 70.0) / 30.0)  # 0 at -70, 1 at -40

        # Migration: follow gradient (electrotaxis)
        migration_dir = grad_V / (np.linalg.norm(grad_V) + 1e-6)

        # Differentiation: threshold for fate
        diff_fate = 1 if V > -50 else (2 if V < -80 else 0)  # Arbitrary types

        # Apoptosis: extreme values
        apoptosis_prob = 1.0 if V > 0 or V < -120 else 0.0

        return {
            'division_rate': division_rate,
            'migration_dir': migration_dir,
            'diff_fate': diff_fate,
            'apoptosis_prob': apoptosis_prob
        }