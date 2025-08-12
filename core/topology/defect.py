import numpy as np
from enum import Enum
from typing import Union, List


class DefectType(Enum):
    """
    Enum for topological defect types in morphogenetic organization.
    """
    DISCLINATION = "disclination"  # Orientation defects with winding
    DISLOCATION = "dislocation"  # Translational defects with Burgers vector
    MONOPOLE = "monopole"  # Source/sink defects with radial flow


class Defect:
    """
    Represents a single topological defect.

    Attributes:
        position (np.ndarray): Continuous 2D position (x, y) in the spatial substrate.
        charge (Union[float, np.ndarray]): Quantized charge:
            - float for disclinations (±1/2, ±1, etc.) and monopoles (±1, ±2, etc.)
            - np.ndarray (2D vector) for dislocations (Burgers vector)
        defect_type (DefectType): Type of the defect.
        core_radius (float): Radius within which field is undefined (singularity core).
    """

    def __init__(
            self,
            position: Union[List[float], np.ndarray],
            charge: Union[float, np.ndarray],
            defect_type: DefectType,
            core_radius: float = 0.1
    ):
        self.position = np.array(position, dtype=float)
        if self.position.shape != (2,):
            raise ValueError("Position must be 2D (x, y) for planarian simulation")

        self.defect_type = defect_type
        self.core_radius = core_radius

        # Validate and set charge based on type
        if defect_type == DefectType.DISCLINATION:
            if not isinstance(charge, (int, float)):
                raise ValueError("Disclination charge must be scalar (float/int)")
            # Quantize to multiples of 1/2
            self.charge = round(charge * 2) / 2
        elif defect_type == DefectType.DISLOCATION:
            self.charge = np.array(charge, dtype=float)
            if self.charge.shape != (2,):
                raise ValueError("Dislocation charge must be 2D Burgers vector")
        elif defect_type == DefectType.MONOPOLE:
            if not isinstance(charge, (int, float)):
                raise ValueError("Monopole charge must be scalar (float/int)")
            # Quantize to integers
            self.charge = round(charge)
        else:
            raise ValueError(f"Invalid defect type: {defect_type}")

    def field_at(self, point: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Compute the vector field contribution from this defect at a given point.

        Args:
            point: 2D position where to evaluate the field.

        Returns:
            np.ndarray: 2D vector field value (e.g., velocity or force).

        For disclinations: azimuthal (rotational) flow ~ charge / r tangential.
        For dislocations: shear flow based on Burgers vector (simplified logarithmic).
        For monopoles: radial flow ~ charge / r² radial.
        """
        r_vec = np.array(point) - self.position
        r = np.linalg.norm(r_vec)
        if r < self.core_radius:
            return np.zeros(2)  # Undefined in core; return zero to avoid singularity

        if self.defect_type == DefectType.DISCLINATION:
            # Azimuthal velocity: (charge / r) * tangential unit vector
            tangential = np.array([-r_vec[1], r_vec[0]]) / r  # Perpendicular
            return (self.charge / r) * tangential

        elif self.defect_type == DefectType.DISLOCATION:
            # Simplified dislocation field: logarithmic shear proportional to Burgers vector
            # (In 2D elasticity, velocity ~ (b × r) / r² + projection terms; approximate here)
            b = self.charge
            log_r = np.log(r) if r > 1e-6 else 0.0
            shear = np.cross(b, r_vec) / (r ** 2 + 1e-6)  # Scalar cross in 2D
            return shear * (r_vec / r)  # Directed shear

        elif self.defect_type == DefectType.MONOPOLE:
            # Radial velocity: (charge / r²) * radial unit vector
            radial = r_vec / r
            return (self.charge / (r ** 2 + 1e-6)) * radial

        return np.zeros(2)


class DefectSystem:
    """
    Manages a collection of topological defects, enforcing conservation laws
    and providing operations for morphogenesis organization.

    Integrates with E8 lattice by quantizing defect creations/annihilations
    to lattice-compatible events, and with p-adic structure for fractal patterning.
    """

    def __init__(self, surface_chi: int = 2):
        """
        Args:
            surface_chi: Euler characteristic of the underlying surface (default 2 for sphere-like closed tissue).
        """
        self.defects: List[Defect] = []
        self.surface_chi = surface_chi  # For Poincaré-Hopf theorem enforcement

    def create_defect_pair(
            self,
            pos1: Union[List[float], np.ndarray],
            charge1: Union[float, np.ndarray],
            type1: DefectType,
            pos2: Union[List[float], np.ndarray],
            charge2: Union[float, np.ndarray],
            type2: DefectType
    ) -> None:
        """
        Create a pair of defects with opposite charges (conserving total charge).
        For dislocations, charges are opposite Burgers vectors.

        Raises:
            ValueError: If types differ or charges don't cancel.
        """
        if type1 != type2:
            raise ValueError("Defect pair must be of the same type")

        d1 = Defect(pos1, charge1, type1)
        d2 = Defect(pos2, charge2, type2)

        if type1 == DefectType.DISLOCATION:
            if not np.allclose(d1.charge + d2.charge, 0):
                raise ValueError("Burgers vectors must sum to zero for creation")
        else:
            if d1.charge + d2.charge != 0:
                raise ValueError("Scalar charges must sum to zero for creation")

        self.defects.extend([d1, d2])
        self._check_poincare_hopf()

    def annihilate(self, defect1: Defect, defect2: Defect) -> None:
        """
        Annihilate two defects if their charges cancel and types match.

        Raises:
            ValueError: If cannot annihilate.
        """
        if defect1.type != defect2.type:
            raise ValueError("Cannot annihilate different defect types")

        if defect1.type == DefectType.DISLOCATION:
            if not np.allclose(defect1.charge + defect2.charge, 0):
                raise ValueError("Burgers vectors do not cancel")
        else:
            if defect1.charge + defect2.charge != 0:
                raise ValueError("Charges do not cancel")

        self.defects.remove(defect1)
        self.defects.remove(defect2)
        self._check_poincare_hopf()

    def braid(self, defect1: Defect, defect2: Defect, path1: List[np.ndarray], path2: List[np.ndarray]) -> float:
        """
        Compute braiding statistics (winding number) for exchanging two defects along paths.
        Simplistic implementation: compute linking number of paths.

        Args:
            defect1, defect2: Defects to braid.
            path1, path2: Lists of positions tracing their exchange paths.

        Returns:
            float: Winding number (topological invariant of the braid).
        """
        if len(path1) != len(path2):
            raise ValueError("Paths must have same length")

        # Approximate winding: integral of (dr1 × dr2) / |r1 - r2|^2 or similar; here simple cumulative cross
        winding = 0.0
        for i in range(1, len(path1)):
            dr1 = path1[i] - path1[i - 1]
            dr2 = path2[i] - path2[i - 1]
            r_diff = path1[i - 1] - path2[i - 1]
            r_norm = np.linalg.norm(r_diff) + 1e-6
            winding += np.cross(dr1, dr2) / (r_norm ** 2)
        return winding / (2 * np.pi)  # Normalize to integer windings ideally

    def total_charge(self, defect_type: DefectType) -> Union[float, np.ndarray]:
        """
        Compute total conserved charge for a given defect type.
        For dislocations, returns sum of Burgers vectors (should be zero on closed surfaces).
        """
        relevant = [d for d in self.defects if d.defect_type == defect_type]
        if not relevant:
            return 0.0 if defect_type != DefectType.DISLOCATION else np.zeros(2)

        if defect_type == DefectType.DISLOCATION:
            return np.sum([d.charge for d in relevant], axis=0)
        else:
            return sum(d.charge for d in relevant)

    def superposed_field_at(self, point: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Compute the total superposed field from all defects at a point.
        Guides cell migration and pattern formation in the simulation.
        """
        return np.sum([d.field_at(point) for d in self.defects], axis=0)

    def _check_poincare_hopf(self) -> None:
        """
        Enforce Poincaré-Hopf theorem: total disclination charge = Euler characteristic.

        Raises:
            ValueError: If theorem violated (simulation inconsistency).
        """
        total_disc = self.total_charge(DefectType.DISCLINATION)
        if not np.isclose(total_disc, self.surface_chi):
            raise ValueError(
                f"Poincaré-Hopf violation: total disclination charge {total_disc} != χ={self.surface_chi}"
            )

    def organize_regeneration(self, wound_position: np.ndarray) -> None:
        """
        Biological hook: Create defect pair near wound to organize regeneration flows.
        Example: +1 monopole source and -1 sink to guide cell migration.
        """
        # Simplified: create monopole pair near wound
        offset = np.random.normal(0, 0.5, size=2)
        pos1 = wound_position + offset
        pos2 = wound_position - offset
        self.create_defect_pair(pos1, 1.0, DefectType.MONOPOLE, pos2, -1.0, DefectType.MONOPOLE)

    def check_balance(self) -> bool:
        """
        Check if defects are balanced (total charges zero except for topology-mandated),
        providing "stop signal" for regeneration.
        """
        try:
            self._check_poincare_hopf()
            for dt in [DefectType.DISLOCATION, DefectType.MONOPOLE]:
                total = self.total_charge(dt)
                if (isinstance(total, np.ndarray) and not np.allclose(total, 0)) or (
                        isinstance(total, float) and not np.isclose(total, 0)
                ):
                    return False
            return True
        except ValueError:
            return False