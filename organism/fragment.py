import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from dynamics.cell import Cell


@dataclass
class Fragment:
    """
    Represents a tissue fragment resulting from amputation, tracking regeneration toward target morphology.

    As per the manuscript, fragments maintain partial morphogenetic information and regenerate based on
    bioelectric patterns, defects, and cellular states. This class encapsulates cells, identifies missing
    regions, and manages regeneration parameters for simulating polarity assessment and healing.

    Attributes:
        cells (Dict[Tuple[int, int], Cell]): Cells by grid position.
        missing_regions (List[str]): Labels for absent anatomical regions (e.g., ['head', 'tail']).
        wound_position (Tuple[int, int]): Primary wound site coordinates.
        regeneration_params (Dict[str, Any]): Parameters guiding regeneration (e.g., {'gradient_direction': np.ndarray}).
    """
    cells: Dict[Tuple[int, int], Cell]
    missing_regions: List[str]
    wound_position: Tuple[int, int]
    regeneration_params: Dict[str, Any]

    def assess_polarity(self) -> str:
        """
        Assesses the fragment's polarity from bioelectric patterns.

        Analyzes voltage gradients along the assumed anterior-posterior (AP) axis to classify as
        'normal' (anterior depolarized), 'reversed', or 'bipolar' (no clear gradient). This informs
        regeneration outcomes, such as head/tail formation in planarian fragments.

        Assumes AP axis along x-coordinate; for general orientations, use regeneration_params['gradient_direction'].

        Returns:
            str: Polarity type ('normal', 'reversed', 'bipolar', or 'undefined' if no cells).
        """
        if not self.cells:
            return 'undefined'

        # Extract positions and voltages
        voltages = [(pos, cell.bioelectric_state.voltage) for pos, cell in self.cells.items()]

        # Sort along x-axis (assumed AP; extendable to arbitrary direction)
        voltages.sort(key=lambda x: x[0][0])  # Sort by x-coordinate

        n = len(voltages)
        # Compute mean voltages in anterior (first third) and posterior (last third)
        anterior_v = np.mean([v for _, v in voltages[:n // 3]]) if n >= 3 else np.mean([v for _, v in voltages])
        posterior_v = np.mean([v for _, v in voltages[-n // 3:]]) if n >= 3 else anterior_v

        if anterior_v > posterior_v + 10:  # Anterior more depolarized (normal gradient)
            return 'normal'
        elif posterior_v > anterior_v + 10:  # Reversed gradient
            return 'reversed'
        else:  # Flat or unclear gradient
            return 'bipolar'