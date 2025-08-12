import numpy as np
from enum import Enum
from typing import Dict, Optional, Tuple, Any, List

from numpy import floating

from core.geometry.e8_lattice import E8Lattice
from core.arithmetic.padic_address import PadicAddress
from core.topology.defect import DefectSystem, DefectType


class CellType(Enum):
    """
    Enum for cell types in the morphogenetic simulation.
    """
    STEM = "stem"  # Pluripotent progenitors
    NEURAL = "neural"  # Nerve tissue
    EPITHELIAL = "epithelial"  # Surface lining
    MUSCLE = "muscle"  # Contractile tissue
    CONNECTIVE = "connective"  # Supportive tissue
    GERM = "germ"  # Reproductive


ALLOWED_TRANSITIONS = {
    CellType.STEM: {CellType.NEURAL, CellType.EPITHELIAL, CellType.MUSCLE, CellType.CONNECTIVE, CellType.GERM}
    # Differentiated types cannot transition further or back to stem (epigenetic barriers)
}


class BioelectricState:
    """
    Container for bioelectric properties.
    """

    def __init__(self, voltage: float = -70.0, current_sensitivity: float = 1.0):
        self.voltage = voltage  # Membrane potential in mV
        self.current_sensitivity = current_sensitivity  # Response to ionic currents


class MechanicalState:
    """
    Container for mechanical properties.
    """

    def __init__(self, pressure: float = 0.0, adhesion: float = 1.0):
        self.pressure = pressure  # Local mechanical pressure
        self.adhesion = adhesion  # Cell-cell adhesion strength


class Cell:
    """
    Represents an individual cell in the morphogenetic simulation.

    Integrates E8 lattice-constrained state, p-adic history, topological awareness,
    and dynamic responses for regeneration and patterning.
    """
    _next_id = 0  # Class-level counter for unique IDs

    def __init__(
            self,
            position: Tuple[int, int],  # Axial hex coordinates (q, r)
            padic_address: PadicAddress,
            cell_type: CellType = CellType.STEM,
            state_vector: Optional[np.ndarray] = None,
            polarity: Optional[np.ndarray] = None,
            age: float = 0.0,
            division_accumulator: float = 0.0,
            division_cooldown: float = 0.0,
            differentiation_timer: float = 0.0,
            migration_velocity: Optional[np.ndarray] = None,
            voltage: float = -70.0,
            current_sensitivity: float = 1.0,
            chemical_receptors: Optional[Dict[str, float]] = None,
            pressure: float = 0.0,
            adhesion: float = 1.0,
            wound_signal_strength: float = 0.0
    ):
        self.unique_id = Cell._next_id
        Cell._next_id += 1

        # Identity
        self.cell_type = cell_type
        self.padic_address = padic_address  # Immutable history
        self.age = age

        # Geometric state
        self.e8 = E8Lattice()
        if state_vector is None:
            state_vector = np.zeros(8)
        self.state_vector = self.e8.project_to_lattice(state_vector)  # Ensure on lattice
        self.position = np.array(position, dtype=int)  # Hex (q, r)
        if polarity is None:
            polarity = np.array([1.0, 0.0])  # Default anterior
        self.polarity = polarity / np.linalg.norm(polarity)  # Unit vector

        # Dynamics
        self.division_accumulator = division_accumulator
        self.division_cooldown = division_cooldown
        self.differentiation_timer = differentiation_timer
        if migration_velocity is None:
            migration_velocity = np.zeros(2)
        self.migration_velocity = migration_velocity

        # Field coupling
        self.bioelectric_state = BioelectricState(voltage, current_sensitivity)
        self.chemical_receptors = chemical_receptors or {}  # e.g., {"Wnt": 1.0, "BMP": 0.5}
        self.mechanical_state = MechanicalState(pressure, adhesion)
        self.wound_signal_strength = wound_signal_strength

    def update(self, dt: float) -> None:
        """
        Integrate cell dynamics over timestep dt.
        Updates age, timers, accumulators; projects state to E8 lattice.
        """
        self.age += dt
        self.division_cooldown = max(0.0, self.division_cooldown - dt)
        self.differentiation_timer = max(0.0, self.differentiation_timer - dt)

        # Example dynamic: accumulate division pressure (placeholder; see compute_pressure)
        self.division_accumulator += dt * self.compute_pressure()

        # Project state vector back to lattice after any perturbations
        self.state_vector = self.e8.project_to_lattice(self.state_vector)

    def apply_fields(self, fields: Dict[str, Any]) -> None:
        """
        Respond to external morphogenetic fields.

        Args:
            fields: Dict with keys like 'bioelectric', 'chemical', 'mechanical', 'wound',
                    each containing field values at cell position.
        """
        if 'bioelectric' in fields:
            electric_field = fields['bioelectric']
            self.bioelectric_state.voltage += self.bioelectric_state.current_sensitivity * electric_field * 0.1  # Simplified update

        if 'chemical' in fields:
            for morphogen, gradient in fields['chemical'].items():
                if morphogen in self.chemical_receptors:
                    response = self.chemical_receptors[morphogen] * gradient
                    self.division_accumulator += response  # Example: morphogens promote division

        if 'mechanical' in fields:
            self.mechanical_state.pressure += fields['mechanical']  # Update pressure

        if 'wound' in fields:
            self.wound_signal_strength = max(self.wound_signal_strength, fields['wound'])

        # Update migration velocity based on fields (gradient following)
        total_force = np.zeros(2)
        if 'chemical' in fields:
            for gradient in fields['chemical'].values():
                total_force += gradient  # Attract to high morphogen
        self.migration_velocity += total_force * 0.01  # Damped update

        # Project state after field application
        self.state_vector = self.e8.project_to_lattice(self.state_vector)

    def check_division(self, space_available: bool, threshold: float = 1.0) -> bool:
        """
        Check if cell is ready to divide.
        Requires accumulator > threshold, cooldown <=0, and space.
        """
        return (
                self.division_accumulator > threshold
                and self.division_cooldown <= 0.0
                and space_available
        )

    def check_death(self, max_age: float = 100.0, max_pressure: float = 10.0) -> bool:
        """
        Check apoptosis conditions: age too high or excessive pressure.
        """
        return self.age > max_age or self.mechanical_state.pressure > max_pressure

    def local_winding(self, neighbors: Dict['Cell', np.ndarray]) -> float:
        """
        Compute local topological charge (winding number) based on neighbor polarities.

        Args:
            neighbors: Dict of neighboring cells to their relative positions.

        Returns:
            float: Approximate winding number (quantized to multiples of 1/2 typically).
        """
        if not neighbors:
            return 0.0

        # Sort neighbors by angle around cell
        angles = [np.arctan2(rel_pos[1], rel_pos[0]) for rel_pos in neighbors.values()]
        sorted_neighbors = sorted(zip(angles, neighbors.keys()), key=lambda x: x[0])

        # Cumulative phase change
        total_wind = 0.0
        prev_pol = sorted_neighbors[0][1].polarity
        for i in range(1, len(sorted_neighbors)):
            curr_pol = sorted_neighbors[i][1].polarity
            delta_theta = np.arctan2(np.cross(prev_pol, curr_pol), np.dot(prev_pol, curr_pol))
            total_wind += delta_theta
            prev_pol = curr_pol

        # Close loop
        delta_theta = np.arctan2(np.cross(prev_pol, sorted_neighbors[0][1].polarity),
                                 np.dot(prev_pol, sorted_neighbors[0][1].polarity))
        total_wind += delta_theta

        return total_wind / (2 * np.pi)  # Winding number

    def neighbor_configuration(self, neighbors: List['Cell']) -> Dict[str, Any]:
        """
        Analyze local topology: count types, average polarity, etc.

        Args:
            neighbors: List of neighboring cells.

        Returns:
            Dict with analysis: {'type_counts': Dict[CellType, int], 'avg_polarity': np.ndarray, ...}
        """
        if not neighbors:
            return {}

        type_counts = {ct: 0 for ct in CellType}
        polarities = []
        for n in neighbors:
            type_counts[n.cell_type] += 1
            polarities.append(n.polarity)

        avg_polarity = np.mean(polarities, axis=0) if polarities else np.zeros(2)

        return {
            'type_counts': type_counts,
            'avg_polarity': avg_polarity,
            'local_density': len(neighbors)
        }

    def defect_distance(self, defect_system: DefectSystem) -> float | floating[Any]:
        """
        Compute distance to nearest topological defect.

        Args:
            defect_system: Global DefectSystem instance.

        Returns:
            float: Euclidean distance to closest defect (inf if none).
        """
        if not defect_system.defects:
            return np.inf

        distances = [np.linalg.norm(self.position - d.position) for d in defect_system.defects]
        return min(distances)

    def compute_pressure(self) -> float:
        """
        Compute tropical pressure for division decision using max-plus algebra.
        Pressure = max(state_deviation, wound_signal, chemical_drive) + local_density

        Returns:
            float: Pressure value driving division accumulator.
        """
        # State deviation from lattice ideal (but since projected, use Casimir or similar)
        state_dev = np.linalg.norm(
            self.state_vector - self.e8.project_to_lattice(self.state_vector + np.random.normal(0, 0.1, 8)))

        # Chemical drive: max over receptors
        chem_drive = max(self.chemical_receptors.values()) if self.chemical_receptors else 0.0

        # Tropical max-plus: max of components, plus additive terms
        tropical_max = max(state_dev, self.wound_signal_strength, chem_drive)
        return tropical_max + self.mechanical_state.pressure  # "Plus" in tropical sense, but here additive for simplicity

    def fate_determination(self, modulus: int = 3) -> None:
        """
        Determine/update cell fate using p-adic address modular arithmetic.
        Only allows transitions from STEM to others.
        """
        if self.cell_type != CellType.STEM:
            return  # No further transitions

        new_type_idx = self.padic_address.cell_type_modular(modulus)
        possible_types = list(ALLOWED_TRANSITIONS[CellType.STEM])
        if new_type_idx < len(possible_types):
            self.cell_type = possible_types[new_type_idx]

    def error_correction(self, golay_code: Optional[np.ndarray] = None) -> None:
        """
        Apply Golay coding for robustness: correct errors in state_vector.
        Placeholder: assume state_vector embeds a 24-bit Golay word (but dim=8; map via hashing or projection).
        Here, simplistic: quantize state to nearest lattice point (built-in projection).

        Args:
            golay_code: Optional external Golay codeword for correction.
        """
        # For now, just re-project to lattice (error correction via quantization)
        self.state_vector = self.e8.project_to_lattice(self.state_vector)

        # If Golay provided, could adjust state_vector components modulo coding
        if golay_code is not None:
            # Example: add small correction based on syndrome (placeholder)
            syndrome = np.sum(self.state_vector) % 2  # Dummy
            if syndrome != 0:
                self.state_vector[0] += 0.5  # Flip a bit-like
            self.state_vector = self.e8.project_to_lattice(self.state_vector)

    def transition_type(self, new_type: CellType) -> bool:
        """
        Attempt type transition; return success.
        Enforces allowed paths.
        """
        if new_type in ALLOWED_TRANSITIONS.get(self.cell_type, set()):
            self.cell_type = new_type
            return True
        return False