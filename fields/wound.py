import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter

from core.geometry.hexagonal_grid import HexagonalGrid
from core.topology.defect import DefectSystem
from dynamics.cell import Cell  # For potential future cell interactions


@dataclass
class WoundSite:
    """
    Represents an individual wound site with dynamic healing properties.

    Tracks position, size, severity, age, and healing status. Severity influences
    initial signal strength and healing duration.
    """
    position: Tuple[int, int]
    radius: float
    severity: float  # 0-1 scale
    age: float = 0.0
    healed: bool = False


class WoundField:
    """
    Manages wound signaling fields that orchestrate regeneration processes.

    As per the manuscript, wounds generate current sources and defect pairs, triggering
    bioelectric depolarization and topological reorganization. This class simulates
    signal propagation with diffusion and decay, integrates with defect systems, and
    modulates healing based on time and potential cell feedback.

    Attributes:
        grid (HexagonalGrid): The underlying hexagonal grid for positions.
        wounds (List[WoundSite]): Active and healing wound sites.
        signal_field (np.ndarray): Wound signal concentration field (flattened; size n_points).
        current_sources (Dict[Tuple[int, int], float]): Injury currents by position.
        healing_rate (float): Base healing rate per time unit.
        signal_decay (float): Exponential decay rate for signals.
        signal_diffusion (float): Diffusion coefficient for signal spread.
    """

    def __init__(self, grid: HexagonalGrid, healing_rate: float = 0.1, signal_decay: float = 0.05,
                 signal_diffusion: float = 0.2):
        """
        Initializes the wound field on a hexagonal grid with healing parameters.

        Args:
            grid (HexagonalGrid): The spatial grid.
            healing_rate (float, optional): Healing progress per time unit. Defaults to 0.1.
            signal_decay (float, optional): Signal decay rate. Defaults to 0.05.
            signal_diffusion (float, optional): Diffusion coefficient. Defaults to 0.2.
        """
        self.grid = grid
        self.n_points = len(grid.positions)
        self.pos_to_idx = {pos: i for i, pos in enumerate(grid.positions)}
        self.idx_to_pos = {i: pos for pos, i in self.pos_to_idx.items()}

        self.wounds: List[WoundSite] = []
        self.signal_field = np.zeros(self.n_points)
        self.current_sources: Dict[Tuple[int, int], float] = {}

        self.healing_rate = healing_rate
        self.signal_decay = signal_decay
        self.signal_diffusion = signal_diffusion

    def create_wound(self, position: Tuple[int, int], radius: float, severity: float = 0.5) -> WoundSite:
        """
        Creates a new wound at the specified position.

        Triggers initial current source and defect creation (via external system).
        Updates signal field immediately.

        Args:
            position (Tuple[int, int]): Wound center on grid.
            radius (float): Initial wound radius.
            severity (float, optional): Severity (0-1). Defaults to 0.5.

        Returns:
            WoundSite: The created wound object.

        Raises:
            ValueError: If position invalid or severity out of range.
        """
        if not self.grid.is_valid_position(position):
            raise ValueError("Invalid position on grid")
        if not 0 <= severity <= 1:
            raise ValueError("Severity must be in [0, 1]")

        wound = WoundSite(position=position, radius=radius, severity=severity)
        self.wounds.append(wound)

        # Create injury current source
        current_magnitude = 10.0 * severity  # μA; scalable
        self.current_sources[position] = current_magnitude

        # Initial signal update
        self._update_signal_field()

        return wound

    def _update_signal_field(self) -> None:
        """
        Updates the signal field from all active wounds.

        Computes radial gradients efficiently per wound, accumulating max strength
        to simulate dominant signal propagation.
        """
        self.signal_field.fill(0.0)
        for wound in self.wounds:
            if wound.healed:
                continue
            wound_cart = self.grid.hex_to_cart(wound.position)
            for idx in range(self.n_points):
                pos = self.idx_to_pos[idx]
                pos_cart = self.grid.hex_to_cart(pos)
                distance = np.linalg.norm(pos_cart - wound_cart)
                if distance <= wound.radius * 3:  # Extended influence
                    strength = wound.severity * np.exp(-distance / wound.radius)
                    self.signal_field[idx] = max(self.signal_field[idx], strength)

    def update(self, dt: float, cells: Optional[Dict[Tuple[int, int], Cell]] = None) -> None:
        """
        Updates wound healing dynamics and signal field.

        Ages wounds, checks healing, modulates radii and currents, applies diffusion
        and decay to signals. Cells can influence healing (placeholder for feedback).

        Args:
            dt (float): Time step.
            cells (Optional[Dict[Tuple[int, int], Cell]]): Cells for potential healing modulation.

        Raises:
            ValueError: If dt <= 0.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        for wound in self.wounds[:]:  # Copy to allow removal
            if not wound.healed:
                wound.age += dt
                # Healing progress (time-based; could modulate with cells nearby)
                healing_progress = wound.age * self.healing_rate / wound.severity
                if healing_progress >= 1.0:
                    wound.healed = True
                    if wound.position in self.current_sources:
                        del self.current_sources[wound.position]
                    self.wounds.remove(wound)  # Optional: clean up healed wounds
                else:
                    # Shrink radius and attenuate current
                    wound.radius *= (1 - self.healing_rate * dt)
                    if wound.position in self.current_sources:
                        self.current_sources[wound.position] *= (1 - self.healing_rate * dt)

        # Update signal field from active wounds
        self._update_signal_field()

        # Apply diffusion (Gaussian approximation)
        sigma = np.sqrt(2 * self.signal_diffusion * dt)
        # Reshape to 2D for filtering (assume rectangular mapping; adjust for hex if needed)
        rows, cols = self.grid.height, self.grid.width  # Assuming rectangular underlying
        signal_2d = self.signal_field.reshape((rows, cols))
        diffused_2d = gaussian_filter(signal_2d, sigma=sigma, mode='constant')
        self.signal_field = diffused_2d.flatten()

        # Apply decay
        self.signal_field *= np.exp(-self.signal_decay * dt)

    def get_wound_influence(self, position: Tuple[int, int]) -> Dict[str, float]:
        """
        Computes wound influence metrics at a given position.

        Includes signal strength, current, and local healing rate if in a healing zone.

        Args:
            position (Tuple[int, int]): Query position.

        Returns:
            Dict[str, float]: {'signal': float, 'current': float, 'healing_rate': float}.

        Raises:
            ValueError: If position invalid.
        """
        if not self.grid.is_valid_position(position):
            raise ValueError("Invalid position")

        idx = self.pos_to_idx[position]
        influence = {
            'signal': self.signal_field[idx],
            'current': self.current_sources.get(position, 0.0),
            'healing_rate': 0.0
        }

        wound_cart = self.grid.hex_to_cart(position)
        for wound in self.wounds:
            if not wound.healed:
                dist = np.linalg.norm(self.grid.hex_to_cart(wound.position) - wound_cart)
                if dist <= wound.radius:
                    influence['healing_rate'] = self.healing_rate * (1 - wound.age / 10.0)  # Age-normalized
                    break  # Assume single dominant wound

        return influence

    def trigger_regeneration_program(self, wound: WoundSite, defect_system: DefectSystem) -> Dict[str, Any]:
        """
        Triggers regeneration at a wound site by creating defects and setting bioelectric patterns.

        As per manuscript, initializes anterior-posterior gradient and organizing defects.

        Args:
            wound (WoundSite): The wound triggering regeneration.
            defect_system (DefectSystem): System to add defects to.

        Returns:
            Dict[str, Any]: Regeneration parameters like voltages and directions.
        """
        # Create defect pair for organization
        defect_system.organize_regeneration(np.array(wound.position))

        # Set up bioelectric gradient parameters
        regeneration_params = {
            'anterior_voltage': -40.0,  # Depolarized head
            'posterior_voltage': -90.0,  # Hyperpolarized tail
            'gradient_direction': np.array([1.0, 0.0]),  # Default along x (A-P axis)
            'organizing_center': wound.position
        }
        return regeneration_params