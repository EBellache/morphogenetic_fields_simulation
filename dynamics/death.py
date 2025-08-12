from typing import Dict, List, Any
import numpy as np

from dynamics.cell import Cell


class ApoptosisMechanism:
    """
    Implements programmed cell death (apoptosis) mechanisms regulated by topological defects and bioelectric thresholds.

    As described in the manuscript, apoptosis is triggered by multiple factors including senescence, extreme bioelectric
    states, mechanical stress, proximity to defect cores, and loss of survival signals. This class evaluates triggers
    and executes death events, recording them for analysis. Such mechanisms are crucial for tissue remodeling and
    preventing aberrant growth during regeneration.

    Attributes:
        death_events (List[Dict[str, Any]]): Record of apoptosis events with details like cause and position.
    """

    def __init__(self):
        """
        Initializes the ApoptosisMechanism with an empty event history.
        """
        self.death_events: List[Dict[str, Any]] = []

    def check_apoptosis_triggers(self, cell: Cell, fields: Dict[str, np.ndarray],
                                 local_topology: Dict[str, float]) -> bool:
        """
        Evaluates multiple apoptosis triggers for a cell.

        Checks conditions in sequence: senescence, voltage extremes, pressure overload, defect proximity,
        and anoikis (loss of adhesion). Returns True if any trigger is activated.

        Args:
            cell (Cell): The cell to evaluate.
            fields (Dict[str, np.ndarray]): External fields, e.g., {'survival_factors': np.ndarray}.
            local_topology (Dict[str, float]): Topology info, e.g., {'defect_distance': 0.3}.

        Returns:
            bool: True if apoptosis should occur, False otherwise.
        """
        if self._check_senescence(cell):
            return True
        if self._check_voltage_death(cell):
            return True
        if self._check_pressure_death(cell):
            return True
        if self._check_defect_death(cell, local_topology):
            return True
        if self._check_anoikis(cell, fields):
            return True
        return False

    def _check_senescence(self, cell: Cell) -> bool:
        """
        Checks for age-related senescence trigger.

        Compares cell age against type-specific maximum lifespan.

        Args:
            cell (Cell): The cell to check.

        Returns:
            bool: True if age exceeds max for type.
        """
        max_age = {
            'STEM': 1000.0,
            'NEURAL': 500.0,
            'EPITHELIAL': 100.0,
            'MUSCLE': 300.0,
            'CONNECTIVE': 200.0
        }.get(cell.cell_type.name, 200.0)
        return cell.age > max_age

    def _check_voltage_death(self, cell: Cell) -> bool:
        """
        Checks for voltage-induced death from extreme depolarization or hyperpolarization.

        As per manuscript, potentials outside [-120, 0] mV are lethal.

        Args:
            cell (Cell): The cell to check.

        Returns:
            bool: True if voltage is extreme.
        """
        V = cell.bioelectric_state.voltage
        return V > 0 or V < -120

    def _check_pressure_death(self, cell: Cell) -> bool:
        """
        Checks for death due to excessive mechanical pressure.

        Args:
            cell (Cell): The cell to check.

        Returns:
            bool: True if pressure > 10.0.
        """
        return cell.mechanical_state.pressure > 10.0

    def _check_defect_death(self, cell: Cell, topology: Dict[str, float]) -> bool:
        """
        Checks for apoptosis induced by proximity to topological defect cores.

        As per manuscript, cells within defect cores (distance < 0.5) undergo programmed death.

        Args:
            cell (Cell): The cell to check (unused; position implicit).
            topology (Dict[str, float]): Must contain 'defect_distance'.

        Returns:
            bool: True if within defect core.

        Raises:
            KeyError: If 'defect_distance' not in topology.
        """
        if 'defect_distance' not in topology:
            raise KeyError("topology must contain 'defect_distance'")
        return topology['defect_distance'] < 0.5

    def _check_anoikis(self, cell: Cell, fields: Dict[str, np.ndarray]) -> bool:
        """
        Checks for anoikis: death from loss of adhesion or survival signals.

        Specific to epithelial cells; triggers if survival factors low or adhesion weak.

        Args:
            cell (Cell): The cell to check.
            fields (Dict[str, np.ndarray]): May contain 'survival_factors' mapped by position.

        Returns:
            bool: True if anoikis conditions met.
        """
        if cell.cell_type.name != 'EPITHELIAL':
            return False
        # Check survival signals if available
        survival = 0.0
        if 'survival_factors' in fields:
            survival_map = fields['survival_factors']
            if isinstance(survival_map, dict) and cell.position in survival_map:
                survival = survival_map[cell.position]
            else:
                survival = 0.0  # Default low if not found
        if survival < 0.1:
            return True
        # Check adhesion strength
        return cell.mechanical_state.adhesion < 0.1

    def execute_apoptosis(self, cell: Cell) -> Dict[str, Any]:
        """
        Executes programmed cell death and records the event.

        Determines cause, creates event dict, and appends to history. In full simulation,
        this could trigger neighbor notifications or field updates (e.g., release of factors).

        Args:
            cell (Cell): The cell undergoing apoptosis.

        Returns:
            Dict[str, Any]: Death event record.
        """
        death_event = {
            'cell_id': cell.unique_id,
            'position': cell.position,
            'type': cell.cell_type.name,
            'age': cell.age,
            'time': cell.age,  # Assuming age tracks time; replace with global time if available
            'cause': self._determine_death_cause(cell)
        }
        self.death_events.append(death_event)
        # Placeholder for side effects (e.g., update fields, notify neighbors)
        return death_event

    def _determine_death_cause(self, cell: Cell) -> str:
        """
        Determines the primary cause of death for event recording.

        Prioritizes checks: senescence > voltage > pressure > unknown.

        Args:
            cell (Cell): The cell.

        Returns:
            str: Cause as string ('senescence', 'voltage_extreme', etc.).
        """
        if cell.age > 500:
            return 'senescence'
        if abs(cell.bioelectric_state.voltage) > 100:
            return 'voltage_extreme'
        if cell.mechanical_state.pressure > 10:
            return 'mechanical_stress'
        return 'unknown'