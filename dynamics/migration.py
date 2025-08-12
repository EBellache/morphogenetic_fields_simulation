import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from core.arithmetic.tropical_algebra import TropicalOperations
from core.geometry.hexagonal_grid import HexagonalGrid
from dynamics.cell import Cell, CellType


@dataclass
class MigrationVector:
    """
    Dataclass representing a cell's migration vector, optimized via tropical algebra.

    Encapsulates direction, speed, mode, and confidence for migration decisions,
    integrating multi-modal cues like chemical and electric gradients.
    """
    direction: np.ndarray  # Unit vector in 2D space
    speed: float  # Migration speed in μm/min
    mode: str  # Migration mode: 'chemotaxis', 'electrotaxis', etc.
    confidence: float  # Confidence in direction (0-1)


class CellMigration:
    """
    Manages cell migration using tropical path optimization on morphogenetic landscapes.

    As per the manuscript, cells follow geodesics in curved spaces defined by fields,
    with cues combined via max-plus algebra for dominant signal selection. This enables
    directed migration toward wounds or gradients while incorporating stochastic exploration.

    Attributes:
        grid_size (Tuple[int, int]): Dimensions of the simulation grid (width, height).
        migration_history (List[Tuple[int, Tuple[int, int], Tuple[int, int]]]): Record of migrations (cell_id, old_pos, new_pos).
    """

    def __init__(self, grid_size: Tuple[int, int]):
        """
        Initializes CellMigration with grid parameters.

        Args:
            grid_size (Tuple[int, int]): Grid dimensions (width, height).
        """
        self.grid_size = grid_size
        self.migration_history: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = []  # (cell_id, old_pos, new_pos)

    def compute_migration_vector(self, cell: Cell, fields: Dict[str, Any], neighbors: List[Cell]) -> MigrationVector:
        """
        Computes the optimal migration vector using tropical geometry.

        Integrates multiple cues (chemotaxis, electrotaxis, etc.) via max-plus selection
        of the dominant vector, with fallback to random walk if cues are weak.

        Args:
            cell (Cell): The migrating cell.
            fields (Dict[str, Any]): External fields, e.g., {'morphogens': Dict[str, np.ndarray(2,)], 'voltage_gradient': np.ndarray(2,)}.
            neighbors (List[Cell]): Neighboring cells for durotaxis computation.

        Returns:
            MigrationVector: Computed vector with direction, speed, mode, and confidence.
        """
        trop = TropicalOperations

        # Compute individual migration cue vectors
        chemotaxis = self._chemotaxis_vector(cell, fields.get('morphogens', {}))
        electrotaxis = self._electrotaxis_vector(cell, fields.get('voltage_gradient'))
        durotaxis = self._durotaxis_vector(cell, neighbors)
        wound_taxis = self._wound_taxis_vector(cell, fields.get('wound_signal'))

        # Collect non-None vectors with modes
        vectors = [
            (chemotaxis, 'chemotaxis') if chemotaxis is not None else None,
            (electrotaxis, 'electrotaxis') if electrotaxis is not None else None,
            (durotaxis, 'durotaxis') if durotaxis is not None else None,
            (wound_taxis, 'wound_taxis') if wound_taxis is not None else None
        ]
        vectors = [v for v in vectors if v is not None]

        if not vectors:
            # No cues: pure random walk
            direction = self._random_walk_vector()
            mode = 'random'
            magnitude = np.linalg.norm(direction)
        else:
            # Tropical selection: dominant vector by max tropical norm (max component after scaling)
            max_trop_norm = float('-inf')
            dominant_vector = np.zeros(2)
            dominant_mode = 'random'
            for vec, mode in vectors:
                # Tropical norm approximation: max(abs(vec)) + log(len(vec)) or similar; here max component
                trop_norm = np.max(np.abs(vec))
                if trop_norm > max_trop_norm:
                    max_trop_norm = trop_norm
                    dominant_vector = vec
                    dominant_mode = mode

            magnitude = np.linalg.norm(dominant_vector)
            if magnitude < 0.1:
                # Weak cue: add random component
                dominant_vector += self._random_walk_vector() * 0.5
                magnitude = np.linalg.norm(dominant_vector)

            direction = dominant_vector / magnitude if magnitude > 0 else np.zeros(2)

        # Compute speed and confidence
        speed = self._compute_speed(cell, dominant_mode)
        confidence = min(1.0, magnitude)

        return MigrationVector(direction=direction, speed=speed, mode=dominant_mode, confidence=confidence)

    def _chemotaxis_vector(self, cell: Cell, morphogens: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Computes chemotactic migration vector from morphogen gradients.

        Sums sensitivity-weighted gradients for multi-morphogen guidance.

        Args:
            cell (Cell): Cell with chemical receptors.
            morphogens (Dict[str, np.ndarray]): Morphogen names to 2D gradient vectors.

        Returns:
            Optional[np.ndarray]: Total chemotaxis vector (2D) or None if no morphogens.
        """
        if not morphogens:
            return None
        total_gradient = np.zeros(2)
        for morphogen, gradient in morphogens.items():
            if morphogen in cell.chemical_receptors and gradient.shape == (2,):
                sensitivity = cell.chemical_receptors[morphogen]
                total_gradient += gradient * sensitivity
        return total_gradient if np.any(total_gradient != 0) else None

    def _electrotaxis_vector(self, cell: Cell, voltage_gradient: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Computes electrotactic migration vector (galvanotaxis).

        Cells migrate toward cathode (negative pole) with type-specific sensitivity.

        Args:
            cell (Cell): Cell with bioelectric properties.
            voltage_gradient (Optional[np.ndarray]): 2D voltage gradient vector.

        Returns:
            Optional[np.ndarray]: Electrotaxis vector (2D) or None if no gradient.
        """
        if voltage_gradient is None or voltage_gradient.shape != (2,):
            return None
        # Type-specific sensitivity factors
        sensitivity = {
            CellType.NEURAL: 2.0,
            CellType.EPITHELIAL: 1.0,
            CellType.STEM: 1.5,
            CellType.MUSCLE: 0.5
        }.get(cell.cell_type, 0.5)
        # Cathodal migration: opposite to gradient (toward lower potential)
        return -voltage_gradient * sensitivity * cell.bioelectric_state.current_sensitivity

    def _durotaxis_vector(self, cell: Cell, neighbors: List[Cell]) -> Optional[np.ndarray]:
        """
        Computes durotactic migration vector toward stiffer regions.

        Averages pressure differences weighted by direction to neighbors.

        Args:
            cell (Cell): Cell with mechanical state.
            neighbors (List[Cell]): Neighbors with positions and pressures.

        Returns:
            Optional[np.ndarray]: Durotaxis vector (2D) or None if no neighbors.
        """
        if not neighbors:
            return None
        gradient = np.zeros(2)
        for neighbor in neighbors:
            dir_vec = np.array(neighbor.position) - np.array(cell.position)
            dist = np.linalg.norm(dir_vec) + 1e-6
            unit_dir = dir_vec / dist
            pressure_diff = neighbor.mechanical_state.pressure - cell.mechanical_state.pressure
            gradient += unit_dir * pressure_diff
        return gradient * 0.5 / len(neighbors) if np.any(gradient != 0) else None  # Average and scale

    def _wound_taxis_vector(self, cell: Cell, wound_signal: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        """
        Computes migration vector toward wound signals for healing response.

        Strength decays with distance; directed toward wound position.

        Args:
            cell (Cell): Migrating cell.
            wound_signal (Optional[Dict[str, Any]]): {'position': Tuple[int, int], 'strength': float}.

        Returns:
            Optional[np.ndarray]: Wound taxis vector (2D) or None if no signal.
        """
        if wound_signal is None:
            return None
        wound_pos = wound_signal.get('position')
        if not isinstance(wound_pos, tuple) or len(wound_pos) != 2:
            return None
        dir_vec = np.array(wound_pos) - np.array(cell.position)
        dist = np.linalg.norm(dir_vec) + 1e-6
        strength = wound_signal.get('strength', 1.0) / dist  # Inverse distance
        return (dir_vec / dist) * strength

    def _random_walk_vector(self) -> np.ndarray:
        """
        Generates a random walk vector for exploratory migration.

        Returns:
            np.ndarray: Unit vector in random direction, scaled mildly.
        """
        angle = np.random.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)]) * 0.1

    def _compute_speed(self, cell: Cell, mode: str) -> float:
        """
        Computes migration speed based on cell type, mode, and age.

        Modulated by local topology as per manuscript (here via mode factor and age slowdown).

        Args:
            cell (Cell): Cell with type and age.
            mode (str): Current migration mode.

        Returns:
            float: Speed in μm/min.
        """
        base_speed = {
            CellType.STEM: 10.0,
            CellType.NEURAL: 15.0,
            CellType.EPITHELIAL: 5.0,
            CellType.MUSCLE: 3.0,
            CellType.CONNECTIVE: 8.0
        }.get(cell.cell_type, 5.0)
        mode_factor = {
            'chemotaxis': 1.2,
            'electrotaxis': 1.5,
            'durotaxis': 0.8,
            'wound_taxis': 2.0,
            'random': 0.5
        }.get(mode, 1.0)
        age_factor = 1.0 / (1 + cell.age / 100.0)  # Slowdown with age
        return base_speed * mode_factor * age_factor

    def execute_migration(self, cell: Cell, migration_vec: MigrationVector, grid: HexagonalGrid, dt: float) -> Tuple[
        int, int]:
        """
        Executes a migration step, updating position if valid.

        Computes new hex position from continuous displacement; checks validity and occupancy
        (occupancy check placeholder). Records migration in history.

        Args:
            cell (Cell): Migrating cell.
            migration_vec (MigrationVector): Computed vector.
            grid (HexagonalGrid): Spatial grid for discretization.
            dt (float): Time step.

        Returns:
            Tuple[int, int]: New position if moved, else original.

        Raises:
            ValueError: If dt <= 0.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        if migration_vec.speed < 0.01:
            return cell.position

        # Compute displacement in continuous space
        displacement = migration_vec.direction * migration_vec.speed * dt

        # Convert current hex to cart, add displacement, convert back to hex
        current_cart = grid.hex_to_cart(cell.position)
        new_cart = current_cart + displacement
        new_hex = grid.cart_to_hex(new_cart)

        # Check validity (and occupancy in full sim)
        if grid.is_valid_position(new_hex):
            # Placeholder: assume unoccupied; in full sim, check and resolve collisions
            old_pos = cell.position
            cell.position = new_hex
            self.migration_history.append((cell.unique_id, old_pos, new_hex))
            return new_hex
        return cell.position  # Blocked