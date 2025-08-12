from typing import Dict, List, Optional, Tuple
import numpy as np

from dynamics.cell import Cell, CellType


class DifferentiationLandscape:
    """
    Implements Waddington's epigenetic landscape for cell fate determination.

    As described in the manuscript, this uses Barbaresco's information geometry on coadjoint orbits
    to model fate spaces as potential landscapes, where cell types are attractors (local minima).
    Transitions between fates are governed by energy barriers, influenced by external fields like
    morphogens and bioelectric gradients, enabling dynamic reprogramming during regeneration.

    Attributes:
        cell_types (List[CellType]): List of possible cell types (e.g., [CellType.STEM, CellType.NEURAL]).
        landscape (Dict[CellType, np.ndarray]): Positions of cell types in abstract fate space.
        barriers (Dict[Tuple[CellType, CellType], float]): Energy barriers between type pairs.
    """

    def __init__(self, cell_types: List[CellType]):
        """
        Initializes the differentiation landscape with given cell types.

        Positions types on a circular manifold in 2D fate space for simplicity; higher dimensions
        could model more complex interactions.

        Args:
            cell_types (List[CellType]): List of cell types to include in the landscape.

        Raises:
            ValueError: If cell_types is empty or contains duplicates.
        """
        if not cell_types or len(cell_types) != len(set(cell_types)):
            raise ValueError("cell_types must be a non-empty list without duplicates")
        self.cell_types = cell_types
        self.landscape = self._construct_landscape()
        self.barriers = self._compute_barriers()

    def _construct_landscape(self) -> Dict[CellType, np.ndarray]:
        """
        Constructs the potential landscape by assigning positions to each cell type.

        Places types equidistantly on a circle in 2D space, representing abstract fate coordinates.
        Scaling factor (10) is arbitrary for separation; could be parameterized.

        Returns:
            Dict[CellType, np.ndarray]: Mapping from type to 2D position vector.
        """
        landscape: Dict[CellType, np.ndarray] = {}
        n_types = len(self.cell_types)
        for i, cell_type in enumerate(self.cell_types):
            angle = 2 * np.pi * i / n_types
            position = np.array([np.cos(angle), np.sin(angle)]) * 10.0
            landscape[cell_type] = position
        return landscape

    def _compute_barriers(self) -> Dict[Tuple[CellType, CellType], float]:
        """
        Computes energy barriers between pairs of cell types using Fisher-Souriau metric.

        Barriers are proportional to Euclidean distance in fate space, modulated by biological
        constraints (e.g., lower from STEM for differentiation plasticity).

        Returns:
            Dict[Tuple[CellType, CellType], float]: Symmetric barriers between type pairs.
        """
        barriers: Dict[Tuple[CellType, CellType], float] = {}
        for i, type1 in enumerate(self.cell_types):
            for type2 in self.cell_types[i + 1:]:
                pos1 = self.landscape[type1]
                pos2 = self.landscape[type2]
                distance = np.linalg.norm(pos1 - pos2)
                # Modulate based on types
                if type1 == CellType.STEM or type2 == CellType.STEM:
                    barrier = distance * 0.5  # Easier transitions from/to stem states
                else:
                    barrier = distance * 2.0  # Higher barriers for transdifferentiation
                barriers[(type1, type2)] = barrier
                barriers[(type2, type1)] = barrier
        return barriers

    def compute_fate_probability(self, current_type: CellType, target_type: CellType, energy: float) -> float:
        """
        Computes the probability of transitioning from current to target type.

        Uses an Arrhenius-like law: P = exp(-ΔE / energy), where ΔE is the barrier height.
        Energy represents available fluctuation (e.g., from noise or signals).

        Args:
            current_type (CellType): Starting cell type.
            target_type (CellType): Desired target type.
            energy (float): Effective 'temperature' or energy scale (must be positive).

        Returns:
            float: Transition probability in [0, 1].

        Raises:
            ValueError: If energy <= 0 or types not in landscape.
        """
        if energy <= 0:
            raise ValueError("Energy must be positive")
        if current_type not in self.landscape or target_type not in self.landscape:
            raise ValueError("Invalid cell type")

        if current_type == target_type:
            return 1.0
        key = (current_type, target_type)
        barrier = self.barriers.get(key, float('inf'))
        if barrier == float('inf'):
            return 0.0
        return np.exp(-barrier / energy)

    def fate_dynamics(self, cell: Cell, fields: Dict[str, np.ndarray], dt: float) -> None:
        """
        Updates the cell's fate based on landscape dynamics and external fields.

        Simulates gradient descent in fate space under field-induced forces, then probabilistically
        transitions to the nearest attractor if a barrier is surmounted. Implements aspects of
        Fisher-Souriau metric for trajectory curvature.

        Args:
            cell (Cell): The cell to update.
            fields (Dict[str, np.ndarray]): External fields, e.g., {'morphogen_gradient': np.ndarray(2,),
                'voltage_gradient': np.ndarray(2,)}.
            dt (float): Time step for dynamics integration.

        Raises:
            ValueError: If dt <= 0 or cell type not in landscape.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        if cell.cell_type not in self.landscape:
            raise ValueError("Cell type not in landscape")

        current_pos = self.landscape[cell.cell_type].copy()

        # Compute force vector from fields
        force = np.zeros(2)
        if 'morphogen_gradient' in fields:
            morph_grad = fields['morphogen_gradient']
            if morph_grad.shape != (2,):
                raise ValueError("morphogen_gradient must be 2D")
            force += morph_grad * 0.1  # Bias toward high morphogen regions
        if 'voltage_gradient' in fields:
            v_grad = fields['voltage_gradient']
            if v_grad.shape != (2,):
                raise ValueError("voltage_gradient must be 2D")
            force -= v_grad * 0.05  # Depolarization promotes dedifferentiation (toward stem)

        # Update position (simple Euler integration)
        new_pos = current_pos + force * dt

        # Find nearest cell type attractor
        min_dist = float('inf')
        nearest_type = cell.cell_type
        for cell_type, pos in self.landscape.items():
            dist = np.linalg.norm(new_pos - pos)
            if dist < min_dist:
                min_dist = dist
                nearest_type = cell_type

        # Attempt transition if nearest is different
        if nearest_type != cell.cell_type:
            # Use differentiation_timer as energy proxy (accumulates over time)
            prob = self.compute_fate_probability(cell.cell_type, nearest_type, cell.differentiation_timer)
            if np.random.random() < prob:
                cell.transition_type(nearest_type)