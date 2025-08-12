import numpy as np
from typing import Callable
from dataclasses import dataclass
from scipy.linalg import expm
from scipy.optimize import fsolve


@dataclass
class ContactStructure:
    """ Contact manifold structure for neurogeometric organization.
    From manuscript: Section 4.1 - Contact geometry underlies the symplectic phase space of morphogenetic dynamics.
    """
    dimension: int  # Must be odd: 2n + 1
    contact_form: np.ndarray  # 1-form α satisfying α ∧ dα^n ≠ 0, shape (dimension,)
    reeb_field: np.ndarray  # Vector field generating contact flow, shape (dimension,)

    def __post_init__(self):
        if self.dimension % 2 == 0:
            raise ValueError("Contact manifold dimension must be odd")
        if self.contact_form.shape != (self.dimension,):
            raise ValueError(f"Contact form must be a 1D array of length {self.dimension}")
        if self.reeb_field.shape != (self.dimension,):
            raise ValueError(f"Reeb field must be a 1D array of length {self.dimension}")

    def contact_bracket(self, f: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Compute contact bracket {f,g} = α(f) g - α(g) f + ω(f, g) Reeb
        Following Souriau's momentum map construction in contact geometry.
        Here, f and g are vector fields in R^{dimension}, ω is the symplectic form from symplectization.
        """
        if f.shape != (self.dimension,) or g.shape != (self.dimension,):
            raise ValueError(f"Vectors must be of dimension {self.dimension}")

        alpha_f = np.dot(self.contact_form, f)
        alpha_g = np.dot(self.contact_form, g)
        omega = self._compute_symplectization()
        # Compute scalar ω(f[:-1], g[:-1]) assuming last coordinate is the contact direction
        # But since omega is 2n x 2n, and dimension=2n+1, we project to the first 2n coordinates
        n = (self.dimension - 1) // 2
        f_symp = f[:2 * n]
        g_symp = g[:2 * n]
        omega_fg = np.dot(f_symp, np.dot(omega, g_symp))  # f^T omega g, scalar

        bracket = alpha_f * g - alpha_g * f + omega_fg * self.reeb_field
        return bracket

    def _compute_symplectization(self) -> np.ndarray:
        """Symplectization: M × R → symplectic manifold with standard form."""
        n = (self.dimension - 1) // 2
        omega = np.zeros((2 * n, 2 * n))
        omega[:n, n:] = np.eye(n)
        omega[n:, :n] = -np.eye(n)
        return omega

    def legendrian_submanifold(self, constraint: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """ Compute Legendrian submanifold where contact form vanishes.
        These are the morphogenetic trajectories.

        Solves α(x) = 0 and constraint(x) = 0, assuming constraint returns a vector of length (dimension - 1).
        Uses numerical solver to find a point on the submanifold.

        Args:
            constraint: Callable that takes x (np.ndarray of shape (dimension,)) and returns np.ndarray of shape (dimension - 1,).

        Returns:
            np.ndarray: A point x satisfying the equations.
        """

        def equations(x):
            alpha_x = np.dot(self.contact_form, x)
            constr_vals = constraint(x)
            return np.append(alpha_x, constr_vals)

        x0 = np.zeros(self.dimension)
        sol = fsolve(equations, x0)
        return sol