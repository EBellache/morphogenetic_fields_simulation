import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from core.arithmetic.padic_address import PadicAddress
from core.geometry.hexagonal_grid import HexagonalGrid
from core.topology.defect import DefectSystem, DefectType, Defect
from core.topology.invariants import IndexTheorem
from dynamics.cell import Cell, CellType
from fields.bioelectric import BioelectricField
from fields.chemical import MorphogenField
from fields.mechanical import MechanicalField
from fields.wound import WoundField, WoundSite
from organism.fragment import Fragment


class PlanarianRegion(Enum):
    """
    Enumeration of anatomical regions in a planarian worm.

    Used for identifying missing parts during regeneration and initializing cell distributions.
    """
    HEAD = "head"
    PHARYNX = "pharynx"
    TAIL = "tail"
    DORSAL = "dorsal"
    VENTRAL = "ventral"


@dataclass
class PlanarianMorphology:
    """
    Defines planarian-specific morphological features and regeneration state.

    As per the manuscript, implements index theorems for regeneration control by tracking
    anatomical markers, axes, and missing regions during fragmentation.

    Attributes:
        length (float): Body length in mm.
        width (float): Body width in mm.
        head_position (Tuple[int, int]): Grid position of head.
        tail_position (Tuple[int, int]): Grid position of tail.
        pharynx_position (Tuple[int, int]): Grid position of pharynx.
        anterior_posterior_axis (np.ndarray): Unit vector for AP axis.
        dorsal_ventral_axis (np.ndarray): Unit vector for DV axis.
        is_fragment (bool): Whether this is a cut fragment.
        missing_regions (List[PlanarianRegion]): Regions to regenerate.
        regenerating (bool): Active regeneration flag.
    """
    length: float = 5.0  # mm
    width: float = 1.0  # mm
    head_position: Tuple[int, int] = (0, 0)
    tail_position: Tuple[int, int] = (100, 30)
    pharynx_position: Tuple[int, int] = (50, 15)
    anterior_posterior_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    dorsal_ventral_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    is_fragment: bool = False
    missing_regions: List[PlanarianRegion] = field(default_factory=list)
    regenerating: bool = False


class Planarian:
    """
    Simulates a complete planarian organism integrating cells, fields, and topology for regeneration.

    Coordinates multi-scale subsystems: cellular dynamics, morphogenetic fields, and topological defects.
    Supports amputation to fragments and checks regeneration completion via index theorems.

    Attributes:
        grid (HexagonalGrid): Spatial grid for the organism.
        morphology (PlanarianMorphology): Current morphological state.
        cells (Dict[Tuple[int, int], Cell]): Cells by grid position.
        bioelectric (BioelectricField): Bioelectric field subsystem.
        morphogens (MorphogenField): Chemical morphogen fields.
        mechanical (MechanicalField): Mechanical stress/strain fields.
        wound_field (WoundField): Wound signaling and healing.
        defect_system (DefectSystem): Topological defect management.
        target_morphology (PlanarianMorphology): Goal state for regeneration.
        regeneration_history (List[Dict[str, Any]]): Log of regeneration events.
    """

    def __init__(self, grid_shape: Tuple[int, int] = (150, 60)):
        """
        Initializes the planarian with a hexagonal grid and standard morphology.

        Sets up cells, fields, and topology subsystems with planarian-specific patterns.

        Args:
            grid_shape (Tuple[int, int]): Grid dimensions (rows, columns).
        """
        self.grid = HexagonalGrid(width=grid_shape[1], height=grid_shape[0])  # Note: (height, width) for consistency
        self.morphology = PlanarianMorphology()
        # Set morphology positions based on grid size (y, x) format for chemical field compatibility
        self.morphology.head_position = (grid_shape[0] // 2, 5)
        self.morphology.tail_position = (grid_shape[0] // 2, grid_shape[1] - 5)  
        self.morphology.pharynx_position = (grid_shape[0] // 2, grid_shape[1] // 2)
        self.cells: Dict[Tuple[int, int], Cell] = {}
        self._initialize_cells()
        self._initialize_fields()
        self._initialize_topology()
        self.target_morphology = PlanarianMorphology()  # Default target: full worm
        self.regeneration_history: List[Dict[str, Any]] = []

    def _initialize_cells(self) -> None:
        """
        Initializes the cell population with region-specific distributions and p-adic addresses.

        Populates head (neural-rich), pharynx (muscle/epithelial), and tail/body (gradient) regions.
        """
        # Head: neural concentration
        head_positions = self._get_region_positions(PlanarianRegion.HEAD)
        for pos in head_positions:
            addr = PadicAddress(prime=3, digits=[0])  # Head lineage
            cell_type = CellType.NEURAL if np.random.random() < 0.7 else CellType.STEM
            self.cells[pos] = Cell(position=pos, padic_address=addr, cell_type=cell_type)

        # Pharynx: muscle and epithelial
        pharynx_positions = self._get_region_positions(PlanarianRegion.PHARYNX)
        for pos in pharynx_positions:
            addr = PadicAddress(prime=3, digits=[1])  # Pharynx lineage
            cell_type = CellType.MUSCLE if np.random.random() < 0.5 else CellType.EPITHELIAL
            self.cells[pos] = Cell(position=pos, padic_address=addr, cell_type=cell_type)

        # Tail and body: mixed with AP gradient
        tail_positions = self._get_region_positions(PlanarianRegion.TAIL)
        for pos in tail_positions:
            addr = PadicAddress(prime=3, digits=[2])  # Tail lineage
            dist_from_head = np.linalg.norm(np.array(pos) - np.array(self.morphology.head_position))
            neural_prob = np.exp(-dist_from_head / 20.0)
            if np.random.random() < neural_prob:
                cell_type = CellType.NEURAL
            elif np.random.random() < 0.3:
                cell_type = CellType.STEM
            else:
                cell_type = CellType.EPITHELIAL
            self.cells[pos] = Cell(position=pos, padic_address=addr, cell_type=cell_type)

    def _initialize_fields(self) -> None:
        """
        Initializes morphogenetic fields with planarian-specific patterns.

        Sets up bioelectric gradients, morphogen sources, mechanical properties, and empty wound field.
        """
        # Bioelectric: AP gradient
        self.bioelectric = BioelectricField(self.grid, D=1.0, tau=1.0, V_rest=-70.0)
        self.bioelectric.set_pattern(self.morphology.head_position, self.morphology.tail_position)

        # Chemical: Wnt posterior, ERK anterior, etc.
        self.morphogens = MorphogenField((self.grid.height, self.grid.width), morphogens=['Wnt', 'ERK', 'FGF', 'BMP'])
        self.morphogens.add_source('ERK', self.morphology.head_position, 10.0)
        self.morphogens.add_source('Wnt', self.morphology.tail_position, 10.0)

        # Mechanical: default properties
        self.mechanical = MechanicalField(self.grid)

        # Wound: initially empty
        self.wound_field = WoundField(self.grid)

    def _initialize_topology(self) -> None:
        """
        Initializes the topological defect system for the planarian (disk topology, χ=1).

        Adds organizing monopoles at head (+1) and tail (-1) for polarity.
        """
        self.defect_system = DefectSystem(surface_chi=1)  # Disk topology
        head_defect = Defect(
            position=np.array(self.morphology.head_position, dtype=float),
            charge=1.0,
            defect_type=DefectType.MONOPOLE
        )
        tail_defect = Defect(
            position=np.array(self.morphology.tail_position, dtype=float),
            charge=-1.0,
            defect_type=DefectType.MONOPOLE
        )
        self.defect_system.defects.extend([head_defect, tail_defect])

    def _get_region_positions(self, region: PlanarianRegion) -> List[Tuple[int, int]]:
        """
        Returns grid positions belonging to a specified anatomical region.

        Defines regions as subsets of the elliptical body: head (anterior 20%), tail (posterior 30%),
        pharynx (central circular area).

        Args:
            region (PlanarianRegion): The anatomical region.

        Returns:
            List[Tuple[int, int]]: Positions within the region and body.
        """
        positions = []
        if region == PlanarianRegion.HEAD:
            end_x = int(self.grid.width * 0.2)
            for x in range(end_x):
                for y in range(self.grid.height):
                    if self._is_inside_body((x, y)):
                        positions.append((x, y))
        elif region == PlanarianRegion.TAIL:
            start_x = int(self.grid.width * 0.7)
            for x in range(start_x, self.grid.width):
                for y in range(self.grid.height):
                    if self._is_inside_body((x, y)):
                        positions.append((x, y))
        elif region == PlanarianRegion.PHARYNX:
            center_x, center_y = self.grid.width // 2, self.grid.height // 2
            radius = 10
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    pos = (center_x + dx, center_y + dy)
                    if self._is_inside_body(pos) and np.sqrt(dx ** 2 + dy ** 2) <= radius:
                        positions.append(pos)
        return positions

    def _is_inside_body(self, position: Tuple[int, int]) -> bool:
        """
        Checks if a position is inside the elliptical planarian body.

        Uses normalized ellipse equation with semi-axes based on grid dimensions.

        Args:
            position (Tuple[int, int]): (x, y) to check.

        Returns:
            bool: True if inside body.
        """
        x, y = position
        center_x, center_y = self.grid.width / 2, self.grid.height / 2
        a = self.grid.width / 2.5  # Semi-major along x
        b = self.grid.height / 4  # Semi-minor along y
        return ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2 <= 1

    def amputate(self, cut_position: int, orientation: str = 'transverse') -> Fragment:
        """
        Performs amputation at a position, creating and returning a fragment.

        Updates morphology to fragment state, creates wound, triggers regeneration,
        and splits cells accordingly. Assumes transverse cuts along x for simplicity.

        Args:
            cut_position (int): x-coordinate for transverse cut.
            orientation (str, optional): Cut type ('transverse'). Defaults to 'transverse'.

        Returns:
            Fragment: The detached fragment.

        Raises:
            ValueError: If cut_position invalid or orientation unsupported.
        """
        if not 0 < cut_position < self.grid.width:
            raise ValueError("cut_position must be within (0, grid width)")
        if orientation != 'transverse':
            raise ValueError("Only 'transverse' orientation supported")

        self.morphology.is_fragment = True
        wound_pos = (cut_position, self.grid.height // 2)

        if cut_position < self.morphology.head_position[0] + 10:
            self.morphology.missing_regions = [PlanarianRegion.HEAD]
            retained_cells = {pos: cell for pos, cell in self.cells.items() if pos[0] >= cut_position}
            fragment_cells = {pos: cell for pos, cell in self.cells.items() if pos[0] < cut_position}
        elif cut_position > self.morphology.tail_position[0] - 10:
            self.morphology.missing_regions = [PlanarianRegion.TAIL]
            retained_cells = {pos: cell for pos, cell in self.cells.items() if pos[0] <= cut_position}
            fragment_cells = {pos: cell for pos, cell in self.cells.items() if pos[0] > cut_position}
        else:
            self.morphology.missing_regions = [PlanarianRegion.HEAD, PlanarianRegion.TAIL]
            retained_cells = {pos: cell for pos, cell in self.cells.items() if
                              cut_position - 20 <= pos[0] <= cut_position + 20}
            # For middle cut, fragment could be the retained or split further; here retain middle
            fragment_cells = {pos: cell for pos, cell in self.cells.items() if
                              pos[0] < cut_position - 20 or pos[0] > cut_position + 20}

        # Create wound and trigger regeneration
        self.wound_field.create_wound(wound_pos, radius=5, severity=0.8)
        regen_params = self.wound_field.trigger_regeneration_program(self.wound_field.wounds[-1], self.defect_system)
        self.morphology.regenerating = True

        # Create fragment
        fragment = Fragment(
            cells=fragment_cells,
            missing_regions=self.morphology.missing_regions.copy(),  # Shared missing for simplicity
            wound_position=wound_pos,
            regeneration_params=regen_params
        )

        # Update own cells
        self.cells = retained_cells
        return fragment

    def check_regeneration_complete(self) -> bool:
        """
        Checks if regeneration is complete using index theorem and morphological criteria.

        Verifies defect charge balance via Poincaré-Hopf and presence of head/tail structures.

        Returns:
            bool: True if complete (stops regenerating).
        """
        theorem = IndexTheorem(surface_genus=0, boundary_components=len(self.wound_field.wounds))
        defect_charges = [d.charge for d in self.defect_system.defects if
                          d.defect_type == DefectType.MONOPOLE]  # Focus on monopoles
        if theorem.check_stopping_condition(defect_charges):
            # Check for head (neural cluster anterior) and tail (posterior cells)
            has_head = any(cell.cell_type == CellType.NEURAL and pos[0] < 20 for pos, cell in self.cells.items())
            has_tail = any(pos[0] > self.grid.width - 20 for pos in self.cells)
            if has_head and has_tail:
                self.morphology.regenerating = False
                self.morphology.is_fragment = False
                self.morphology.missing_regions = []
                return True
        return False