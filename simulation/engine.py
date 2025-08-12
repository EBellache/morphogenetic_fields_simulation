import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

from core.coding.triple_code import TripleCode
from core.geometry.hexagonal_grid import HexagonalGrid
from core.topology.invariants import IndexTheorem
from dynamics.cell import Cell, CellType
from dynamics.death import ApoptosisMechanism
from dynamics.differentiation import DifferentiationLandscape
from dynamics.division import CellDivisionMechanics, DivisionMode
from dynamics.migration import CellMigration, MigrationVector
from organism.planarian import Planarian


@dataclass
class SimulationConfig:
    """
    Configuration parameters for the morphogenetic simulation.

    Defines grid dimensions, time steps, field properties, cell behaviors, and checkpointing.
    These settings control the fidelity and duration of regeneration simulations.
    """
    grid_shape: Tuple[int, int] = (150, 60)  # (height, width) for hexagonal grid
    dt: float = 0.01  # Time step in arbitrary units (e.g., minutes)
    total_time: float = 100.0  # Total simulation duration

    # Field parameters
    voltage_diffusion: float = 1.0  # D for bioelectric diffusion
    morphogen_diffusion: float = 0.1  # D for chemical diffusion
    mechanical_viscosity: float = 1.0  # η for viscous stress

    # Cell parameters
    division_threshold: float = 1.0  # Tropical pressure for division
    migration_speed: float = 5.0  # Base speed in μm/min

    # Topology parameters
    defect_mobility: float = 0.1  # Defect drift speed factor

    # Triple code parameters
    golay_error_rate: float = 0.01  # Simulated error probability
    e8_projection_strength: float = 0.5  # Lattice snap-back strength
    ag_coordination_range: float = 10.0  # AG code influence radius

    # Checkpointing
    checkpoint_interval: float = 10.0  # Time between checkpoints
    save_snapshots: bool = True  # Whether to save state snapshots


class MorphogeneticSimulation:
    """
    Main engine for simulating morphogenesis using the triple-code framework.

    Coordinates the Planarian organism with cellular dynamics managers, fields, topology,
    and error correction. Implements time-stepping, experimental protocols, checkpoints,
    and stopping conditions based on topological invariants.

    Attributes:
        config (SimulationConfig): Simulation parameters.
        time (float): Current simulation time.
        step_count (int): Number of steps executed.
        organism (Planarian): The simulated planarian instance.
        triple_code (TripleCode): Integrated coding system for robustness.
        division_manager (CellDivisionMechanics): Handles cell divisions.
        differentiation_landscape (DifferentiationLandscape): Manages fate transitions.
        migration_manager (CellMigration): Controls cell movements.
        death_manager (ApoptosisMechanism): Processes apoptosis.
        history (List[Dict[str, Any]]): Snapshots of simulation state.
        metrics (Dict[str, Any]): Performance and outcome metrics.
        stable_steps (int): Consecutive steps with stable invariants (for stopping).
    """

    def __init__(self, config: SimulationConfig):
        """
        Initializes the simulation with configuration and subsystems.

        Args:
            config (SimulationConfig): Parameters for the run.
        """
        self.config = config
        self.time = 0.0
        self.step_count = 0
        self.stable_steps = 0

        # Initialize organism
        self.organism = Planarian(grid_shape=config.grid_shape)
        self.grid = self.organism.grid  # Reference to grid for convenience

        # Initialize triple code
        self.triple_code = TripleCode()

        # Initialize dynamics managers
        self.division_manager = CellDivisionMechanics(self.organism.grid, division_threshold=config.division_threshold)
        self.differentiation_landscape = DifferentiationLandscape([ct for ct in CellType])
        self.migration_manager = CellMigration(config.grid_shape)
        self.death_manager = ApoptosisMechanism()

        # State tracking
        self.history: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}

    def run(self, experiment_protocol: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the complete simulation with optional experimental protocol.

        Applies protocol, runs time steps until total_time or stopping condition,
        computes metrics, and returns results including final state and history.

        Args:
            experiment_protocol (Optional[Dict[str, Any]]): Dict defining experiments like amputations.

        Returns:
            Dict[str, Any]: {'final_state': Dict, 'metrics': Dict, 'history': List}.
        """
        start_time = time.time()

        # Apply initial experimental protocol if provided
        if experiment_protocol:
            self._apply_protocol(experiment_protocol)

        # Main loop
        while self.time < self.config.total_time:
            self.step()
            if self.step_count % int(self.config.checkpoint_interval / self.config.dt) == 0:
                self._checkpoint()
            if self._check_stopping_conditions():
                break

        # Compute final metrics
        self.metrics = self._compute_metrics()
        self.metrics['simulation_time'] = time.time() - start_time

        return {
            'final_state': self._get_state_snapshot(),
            'metrics': self.metrics,
            'history': self.history if self.config.save_snapshots else []
        }

    def step(self) -> None:
        """
        Advances the simulation by one time step.

        Updates fields, cells, processes decisions (division/migration/death), topology,
        error correction, and regeneration if active.
        """
        dt = self.config.dt

        # 1. Update fields
        self._update_fields(dt)

        # 2. Update cell states
        self._update_cells(dt)

        # 3. Process cell decisions
        self._process_cell_decisions(dt)

        # 4. Update topology
        self._update_topology(dt)

        # 5. Apply triple-code error correction
        self._apply_error_correction()

        # 6. Update regeneration if active
        if self.organism.morphology.regenerating:
            self._update_regeneration(dt)

        # Increment time
        self.time += dt
        self.step_count += 1

    def _update_fields(self, dt: float) -> None:
        """
        Updates all morphogenetic fields for the time step.

        Includes bioelectric, chemical, mechanical, and wound fields.
        """
        self.organism.bioelectric.update(dt, self.organism.cells)
        self.organism.morphogens.update(dt, self.organism.cells)
        self.organism.mechanical.update(dt, self.organism.cells)
        self.organism.wound_field.update(dt, self.organism.cells)

    def _update_cells(self, dt: float) -> None:
        """
        Updates individual cell states by applying local field values.
        """
        for pos, cell in self.organism.cells.items():
            fields = self._gather_fields_at(pos)
            cell.apply_fields(fields)
            cell.update(dt)

    def _process_cell_decisions(self, dt: float) -> None:
        """
        Processes division, migration, and death decisions for all cells.

        Handles removals and additions carefully to avoid modifying dict during iteration.
        """
        cells_to_remove: List[Tuple[int, int]] = []
        cells_to_add: Dict[Tuple[int, int], Cell] = {}

        for pos, cell in list(self.organism.cells.items()):
            neighbors_pos = self.grid.get_neighbors(pos)
            neighbors = [self.organism.cells.get(p) for p in neighbors_pos if p in self.organism.cells]

            # Check death
            local_topology = {'defect_distance': cell.defect_distance(self.organism.defect_system)}
            fields = self._gather_fields_at(pos)  # For anoikis, etc.
            if self.death_manager.check_apoptosis_triggers(cell, fields, local_topology):
                cells_to_remove.append(pos)
                continue

            # Check division
            pressure = self.division_manager.compute_division_pressure(cell, neighbors, fields)
            space_available = len(neighbors) < 6  # Hex max neighbors
            if cell.check_division(space_available, self.config.division_threshold):
                pressure_components = {
                    'mechanical': cell.mechanical_state.pressure,
                    'bioelectric': max(0, (cell.bioelectric_state.voltage + 70) / 30),
                    'wound': cell.wound_signal_strength
                }
                mode = self.division_manager.select_division_mode(cell, pressure_components, local_topology)
                defect_influence = local_topology['defect_distance'] if 'defect_distance' in local_topology else 0.0
                bioelectric_bias = fields.get('bioelectric', 0.0)
                daughters = self.division_manager.execute_division(
                    cell, mode, self.grid, pressure, defect_influence, bioelectric_bias
                )
                if daughters:
                    d1, d2 = daughters
                    cells_to_add[d1.position] = d1
                    cells_to_add[d2.position] = d2
                    cells_to_remove.append(pos)

            # Check migration
            migration_vec = self.migration_manager.compute_migration_vector(cell, fields, neighbors)
            if migration_vec.speed > 0.01:
                new_pos = self.migration_manager.execute_migration(cell, migration_vec, self.grid, dt)
                if new_pos != pos and new_pos not in self.organism.cells and new_pos not in cells_to_add:
                    cells_to_add[new_pos] = cell
                    cells_to_remove.append(pos)

        # Apply changes
        for pos in cells_to_remove:
            if pos in self.organism.cells:
                del self.organism.cells[pos]
        for pos, new_cell in cells_to_add.items():
            self.organism.cells[pos] = new_cell
            new_cell.position = pos  # Ensure position updated

    def _update_topology(self, dt: float) -> None:
        """
        Updates topological defects: drifts along gradients and checks annihilations.
        """
        # Drift defects along local voltage gradients
        for defect in self.organism.defect_system.defects:
            nearest_pos = self.grid.closest_position(defect.position)
            grad_V = self.organism.bioelectric.get_gradient_at(nearest_pos)
            velocity = self.config.defect_mobility * grad_V
            defect.position += velocity * dt

        # Check for annihilations (pairwise distance <1 and charges cancel)
        to_remove = []
        defects = self.organism.defect_system.defects
        for i in range(len(defects)):
            for j in range(i + 1, len(defects)):
                d1, d2 = defects[i], defects[j]
                if d1.defect_type == d2.defect_type and np.linalg.norm(d1.position - d2.position) < 1.0:
                    if isinstance(d1.charge, (int, float)) and abs(d1.charge + d2.charge) < 0.1:
                        to_remove.extend([i, j])
        for idx in sorted(set(to_remove), reverse=True):
            del defects[idx]

    def _apply_error_correction(self) -> None:
        """
        Applies triple-code error correction to all cells for robustness.
        """
        for pos, cell in self.organism.cells.items():
            cell_state = {
                'local_bits': np.random.randint(0, 2, 12),  # Placeholder; derive from cell state
                'continuous_state': cell.state_vector,
                'position': pos
            }
            encoded = self.triple_code.encode_state(cell_state)
            corrected = self.triple_code.error_correct(encoded)
            if 'e8' in corrected:
                cell.state_vector = corrected['e8']

    def _update_regeneration(self, dt: float) -> None:
        """
        Updates ongoing regeneration processes if active.

        Placeholder for orchestration; in full sim, could call a RegenerationOrchestrator.
        """
        # Assuming a orchestrator class; implement as needed
        pass

    def _check_stopping_conditions(self) -> bool:
        """
        Checks global stopping conditions based on topological invariants.

        Halts if invariants balanced for sufficient stable steps.

        Returns:
            bool: True if simulation should stop.
        """
        theorem = IndexTheorem()
        defect_charges = [d.charge for d in self.organism.defect_system.defects if isinstance(d.charge, (int, float))]
        if theorem.check_stopping_condition(defect_charges):
            self.stable_steps += 1
            if self.stable_steps > 100:
                return True
        else:
            self.stable_steps = 0
        return False

    def _gather_fields_at(self, position: Tuple[int, int]) -> Dict[str, Any]:
        """
        Gathers local field values at a position for cell updates.

        Returns:
            Dict[str, Any]: Fields like {'voltage': float, 'morphogens': Dict[str, float], ...}.
        """
        # Handle coordinate conversion for field access
        if isinstance(position, tuple) and len(position) == 2:
            y, x = position if position[0] < self.organism.morphogens.grid_shape[0] else (position[1], position[0])
            y = max(0, min(y, self.organism.morphogens.grid_shape[0] - 1))
            x = max(0, min(x, self.organism.morphogens.grid_shape[1] - 1))
        else:
            y, x = 0, 0
            
        fields = {
            'voltage': self.organism.bioelectric.voltage.get(position, -70.0) if hasattr(self.organism.bioelectric.voltage, 'get') else -70.0,
            'morphogens': {m: float(self.organism.morphogens.fields[m][y, x]) for m in self.organism.morphogens.morphogens},
            'mechanical': self.organism.mechanical.pressure.get(position, 0.0) if hasattr(self.organism.mechanical.pressure, 'get') else 0.0,
            'defects': self.organism.defect_system.defects,
            'wound_signal': self.organism.wound_field.get_wound_influence(position)['signal']
        }
        # Add gradients if needed
        fields['voltage_gradient'] = self.organism.bioelectric.get_gradient_at(position)
        return fields

    def _apply_protocol(self, protocol: Dict[str, Any]) -> None:
        """
        Applies an experimental protocol like amputations or perturbations.

        Args:
            protocol (Dict[str, Any]): Experiment specs, e.g., {'amputation': {'position': 50}}.
        """
        if 'amputation' in protocol:
            cut_pos = protocol['amputation']['position']
            self.organism.amputate(cut_pos)
        if 'voltage_perturbation' in protocol:
            pos = protocol['voltage_perturbation']['position']
            delta_v = protocol['voltage_perturbation']['magnitude']
            self.organism.bioelectric.perturb(pos, delta_v)
        if 'morphogen_injection' in protocol:
            morphogen = protocol['morphogen_injection']['type']
            pos = protocol['morphogen_injection']['position']
            amount = protocol['morphogen_injection']['amount']
            self.organism.morphogens.add_source(morphogen, pos, amount)

    def _checkpoint(self) -> None:
        """
        Saves a checkpoint of the current state if configured.
        """
        if self.config.save_snapshots:
            snapshot = self._get_state_snapshot()
            self.history.append(snapshot)

    def _get_state_snapshot(self) -> Dict[str, Any]:
        """
        Captures a snapshot of the current simulation state.

        Returns:
            Dict[str, Any]: State info like cell count, defects, etc.
        """
        return {
            'time': self.time,
            'cell_count': len(self.organism.cells),
            'cell_types': self._count_cell_types(),
            'defect_count': len(self.organism.defect_system.defects),
            'total_charge': sum(
                d.charge for d in self.organism.defect_system.defects if isinstance(d.charge, (int, float))),
            'regenerating': self.organism.morphology.regenerating,
            'wounds': sum(1 for w in self.organism.wound_field.wounds if not w.healed)
        }

    def _count_cell_types(self) -> Dict[str, int]:
        """
        Counts cells by type in the current organism.

        Returns:
            Dict[str, int]: Type name to count.
        """
        counts: Dict[str, int] = {}
        for cell in self.organism.cells.values():
            typ = cell.cell_type.name
            counts[typ] = counts.get(typ, 0) + 1
        return counts

    def _compute_metrics(self) -> Dict[str, Any]:
        """
        Computes final simulation metrics like cell counts and defect balance.

        Returns:
            Dict[str, Any]: Metrics dictionary.
        """
        return {
            'final_cell_count': len(self.organism.cells),
            'cell_type_distribution': self._count_cell_types(),
            'total_divisions': len(self.division_manager.division_history),
            'total_deaths': len(self.death_manager.death_events),
            'final_defect_balance': abs(sum(d.charge for d in self.organism.defect_system.defects if
                                            isinstance(d.charge,
                                                       (int, float))) - self.organism.defect_system.surface_chi),
            'regeneration_complete': not self.organism.morphology.regenerating,
            'simulation_steps': self.step_count
        }