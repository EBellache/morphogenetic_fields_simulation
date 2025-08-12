import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from core.arithmetic.tropical_algebra import TropicalOperations
from core.arithmetic.padic_address import PadicAddress
from core.geometry.hexagonal_grid import HexagonalGrid
from core.topology.defect import Defect, DefectType
from dynamics.cell import Cell, CellType


class DivisionMode(Enum):
    """
    Enumeration of cell division modes, determined by tropical pressure and local defect configurations.

    These modes reflect different biological division strategies observed in morphogenesis,
    such as symmetric proliferation or asymmetric stem cell differentiation.
    """
    SYMMETRIC = "symmetric"  # Equal daughter cells, common in tissue expansion
    ASYMMETRIC = "asymmetric"  # One stem, one differentiated, for lineage commitment
    SPIRAL = "spiral"  # Chiral division influenced by orientational defects
    PLANAR = "planar"  # Division confined to tissue plane, e.g., in epithelia


@dataclass
class DivisionEvent:
    """
    Dataclass recording a cell division event for lineage tracking and analysis.

    Captures key parameters like mode, timing, and influencing factors for post-simulation review.
    """
    parent_id: int
    daughter_ids: Tuple[int, int]
    mode: DivisionMode
    time: float
    position: Tuple[int, int]
    tropical_pressure: float
    defect_influence: float
    bioelectric_bias: float


class CellDivisionMechanics:
    """
    Manages cell division mechanics using tropical optimization and multi-field integration.

    As described in the manuscript, division is driven by max-plus pressure computations,
    incorporating mechanical, bioelectric, chemical, and topological influences. This class
    computes division readiness, selects modes, and executes divisions while maintaining
    p-adic developmental histories.

    Attributes:
        grid_size (Tuple[int, int]): Dimensions of the simulation grid (width, height).
        division_history (List[DivisionEvent]): Record of all division events.
        division_threshold (float): Tropical pressure threshold for triggering division.
    """

    def __init__(self, grid_size: Tuple[int, int], division_threshold: float = 1.0):
        """
        Initializes the CellDivisionMechanics with grid parameters and threshold.

        Args:
            grid_size (Tuple[int, int]): Grid dimensions (width, height).
            division_threshold (float, optional): Pressure threshold for division. Defaults to 1.0.
        """
        self.grid_size = grid_size
        self.division_history: List[DivisionEvent] = []
        self.division_threshold = division_threshold

    def compute_division_pressure(self, cell: Cell, neighbors: List[Cell], fields: Dict[str, Any]) -> float:
        """
        Computes the tropical (max-plus) pressure driving cell division.

        Pressure is the tropical maximum of component pressures (mechanical, bioelectric,
        chemical, wound), augmented by defect contributions. From manuscript: pressure =
        max(state_deviation, wound_signal, chemical_drive) with tropical operations.

        Args:
            cell (Cell): The cell under consideration.
            neighbors (List[Cell]): List of neighboring cells for local interactions.
            fields (Dict[str, Any]): External fields, e.g., {'voltage': np.ndarray, 'morphogens': Dict, 'wound_signal': float, 'defects': List[Defect]}.

        Returns:
            float: Computed tropical pressure value.
        """
        trop = TropicalOperations

        # Compute individual pressure components
        mechanical = self._mechanical_pressure(cell, neighbors)
        bioelectric = self._bioelectric_pressure(cell, fields.get('voltage'))
        chemical = self._chemical_pressure(cell, fields.get('morphogens', {}))
        wound = fields.get('wound_signal', 0.0)

        # Tropical max over base components
        pressure = trop.tropical_add(trop.tropical_add(mechanical, bioelectric), trop.tropical_add(chemical, wound))

        # Add defect contribution if available
        if 'defects' in fields:
            defect_term = self._defect_contribution(cell.position, fields['defects'])
            pressure = trop.tropical_add(pressure, defect_term)

        return pressure

    def _mechanical_pressure(self, cell: Cell, neighbors: List[Cell]) -> float:
        """
        Computes mechanical pressure from crowding and adhesion variance.

        Higher neighbor count and adhesion heterogeneity increase pressure.

        Args:
            cell (Cell): The cell (unused here but included for consistency).
            neighbors (List[Cell]): Neighboring cells.

        Returns:
            float: Mechanical pressure value.
        """
        if not neighbors:
            return 0.0
        crowding = len(neighbors) / 6.0  # Normalize by max hex neighbors
        adhesion_values = [n.mechanical_state.adhesion for n in neighbors]
        adhesion_variance = np.var(adhesion_values) if adhesion_values else 0.0
        return crowding + 0.5 * adhesion_variance

    def _bioelectric_pressure(self, cell: Cell, voltage_field: Optional[np.ndarray]) -> float:
        """
        Computes bioelectric pressure from membrane depolarization.

        Depolarization above resting potential promotes division.

        Args:
            cell (Cell): The cell with bioelectric state.
            voltage_field (Optional[np.ndarray]): External voltage field (unused; uses cell's voltage).

        Returns:
            float: Bioelectric pressure value.
        """
        V_rest = -70.0  # Typical resting potential in mV
        V = cell.bioelectric_state.voltage
        if V > V_rest:
            return (V - V_rest) / 30.0  # Normalize to ~1 at -40 mV
        return 0.0

    def _chemical_pressure(self, cell: Cell, morphogens: Dict[str, float]) -> float:
        """
        Computes chemical pressure from morphogen levels.

        Mitogenic morphogens (e.g., Wnt) increase pressure; differentiation signals (e.g., BMP) decrease it.

        Args:
            cell (Cell): The cell (unused here).
            morphogens (Dict[str, float]): Morphogen concentrations, e.g., {'Wnt': 1.2, 'BMP': 0.5}.

        Returns:
            float: Chemical pressure value (non-negative).
        """
        if not morphogens:
            return 0.0
        pressure = 0.0
        if 'Wnt' in morphogens:
            pressure += morphogens['Wnt'] * 0.8
        if 'BMP' in morphogens:
            pressure -= morphogens['BMP'] * 0.3
        return max(0.0, pressure)

    def _defect_contribution(self, position: Tuple[int, int], defects: List[Defect]) -> float:
        """
        Computes pressure contribution from nearby topological defects.

        Positive disclinations and monopole sources promote division within an influence radius.

        Args:
            position (Tuple[int, int]): Cell position.
            defects (List[Defect]): List of defects in the system.

        Returns:
            float: Defect-induced pressure term.
        """
        contribution = 0.0
        pos_array = np.array(position)
        for defect in defects:
            dist = np.linalg.norm(pos_array - defect.position)
            if dist < 5.0:  # Arbitrary influence radius
                if defect.defect_type == DefectType.DISCLINATION and defect.charge > 0:
                    contribution += defect.charge / (1 + dist)
                elif defect.defect_type == DefectType.MONOPOLE and defect.charge > 0:
                    contribution += defect.charge / (1 + dist ** 2)
        return contribution

    def select_division_mode(self, cell: Cell, pressure_components: Dict[str, float],
                             local_topology: Dict[str, float]) -> DivisionMode:
        """
        Selects the division mode based on pressure components and local topology.

        High wound signals favor asymmetric division; proximity to positive defects favors spiral;
        epithelial cells prefer planar. Defaults to symmetric.

        Args:
            cell (Cell): The dividing cell.
            pressure_components (Dict[str, float]): Breakdown of pressure sources, e.g., {'wound': 0.6}.
            local_topology (Dict[str, float]): Topology info, e.g., {'nearest_defect_charge': 0.5}.

        Returns:
            DivisionMode: Selected mode for the division.
        """
        if pressure_components.get('wound', 0.0) > 0.5:
            return DivisionMode.ASYMMETRIC
        if local_topology.get('nearest_defect_charge', 0.0) > 0.4:
            return DivisionMode.SPIRAL
        if cell.cell_type == CellType.EPITHELIAL:
            return DivisionMode.PLANAR
        return DivisionMode.SYMMETRIC

    def execute_division(self, parent_cell: Cell, mode: DivisionMode, grid: HexagonalGrid, tropical_pressure: float,
                         defect_influence: float, bioelectric_bias: float) -> Optional[Tuple[Cell, Cell]]:
        """
        Executes the cell division, creating and placing daughter cells.

        Implements p-adic branching for developmental histories and state partitioning based on mode.
        Records the event in history. Returns None if insufficient space.

        Args:
            parent_cell (Cell): The parent cell dividing.
            mode (DivisionMode): Selected division mode.
            grid (HexagonalGrid): The spatial grid for placement.
            tropical_pressure (float): Computed pressure for the event record.
            defect_influence (float): Defect contribution for the record.
            bioelectric_bias (float): Bioelectric bias for the record.

        Returns:
            Optional[Tuple[Cell, Cell]]: Daughter cells if successful, else None.
        """
        pos = parent_cell.position
        available = self._find_division_sites(pos, grid)
        if len(available) < 2:
            return None  # Insufficient space

        # Branch p-adic addresses (binary branching for simplicity)
        daughter1_addr = parent_cell.padic_address.branch(0)
        daughter2_addr = parent_cell.padic_address.branch(1)

        # Determine types and state split
        if mode == DivisionMode.ASYMMETRIC:
            d1_type = parent_cell.cell_type  # Stem
            d2_type = self._select_differentiated_type(parent_cell)
            state_split = 0.7  # Stem retains more state
        else:
            d1_type = d2_type = parent_cell.cell_type
            state_split = 0.5

        # Create daughters
        daughter1 = self._create_daughter(parent_cell, available[0], daughter1_addr, d1_type, state_split)
        daughter2 = self._create_daughter(parent_cell, available[1], daughter2_addr, d2_type, 1 - state_split)

        # Record event
        event = DivisionEvent(
            parent_id=parent_cell.unique_id,
            daughter_ids=(daughter1.unique_id, daughter2.unique_id),
            mode=mode,
            time=parent_cell.age,
            position=pos,
            tropical_pressure=tropical_pressure,
            defect_influence=defect_influence,
            bioelectric_bias=bioelectric_bias
        )
        self.division_history.append(event)

        return daughter1, daughter2

    def _find_division_sites(self, position: Tuple[int, int], grid: HexagonalGrid) -> List[Tuple[int, int]]:
        """
        Finds available unoccupied positions for daughter cells adjacent to parent.

        Args:
            position (Tuple[int, int]): Parent position.
            grid (HexagonalGrid): The grid (assumes get_neighbors returns all adjacent).

        Returns:
            List[Tuple[int, int]]: Up to 2 available sites (filter for unoccupied in full sim).
        """
        neighbors = grid.get_neighbors(position)
        # In full simulation, filter for unoccupied; here return first 2 for simplicity
        return neighbors[:2] if len(neighbors) >= 2 else []

    def _select_differentiated_type(self, parent_cell: Cell) -> CellType:
        """
        Selects differentiated type for asymmetric division using p-adic modular arithmetic.

        Maps modular residue to cell types (e.g., 0→NEURAL, 1→EPITHELIAL, 2→MUSCLE).

        Args:
            parent_cell (Cell): Parent with p-adic address.

        Returns:
            CellType: Selected differentiated type.
        """
        fate_index = parent_cell.padic_address.cell_type_modular(3)
        fate_map = {0: CellType.NEURAL, 1: CellType.EPITHELIAL, 2: CellType.MUSCLE}
        return fate_map.get(fate_index, CellType.EPITHELIAL)

    def _create_daughter(self, parent: Cell, position: Tuple[int, int], address: PadicAddress, cell_type: CellType,
                         state_fraction: float) -> Cell:
        """
        Creates a daughter cell with inherited and modified properties.

        Copies parent attributes, scales state vector, resets age, etc.

        Args:
            parent (Cell): Parent cell.
            position (Tuple[int, int]): New position for daughter.
            address (PadicAddress): Branched p-adic address.
            cell_type (CellType): Assigned cell type.
            state_fraction (float): Fraction of parent state vector to inherit.

        Returns:
            Cell: New daughter cell instance.
        """
        # Create new Cell with inherited properties (simplified; copy relevant attrs)
        daughter = Cell(
            position=position,
            padic_address=address,
            cell_type=cell_type,
            state_vector=parent.state_vector * state_fraction,
            polarity=parent.polarity.copy(),  # Inherit orientation
            age=0.0,
            division_accumulator=0.0,
            division_cooldown=parent.division_cooldown,  # Inherit cooldown?
            differentiation_timer=0.0,
            migration_velocity=np.zeros(2),
            voltage=parent.bioelectric_state.voltage,  # Inherit potential
            current_sensitivity=parent.bioelectric_state.current_sensitivity,
            chemical_receptors=parent.chemical_receptors.copy(),
            pressure=parent.mechanical_state.pressure * state_fraction,
            adhesion=parent.mechanical_state.adhesion,
            wound_signal_strength=0.0  # Reset wound signal
        )
        return daughter