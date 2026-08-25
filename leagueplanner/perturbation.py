import numpy as np

from .constants import LARGE_NBR


class Perturbation:
    """Helper class to modify current schedule to avoid local optima."""

    def __init__(self, alpha: float = 0.50, beta: float = 0.01) -> None:
        """
        Initializes a new instance of the Perturbation class.

        See PlannerParams for parameter details.
        """
        self.alpha = alpha
        self.beta = beta

    def perturbate(self, X: np.ndarray) -> None:
        """Perturbates the given matrix in-place with the first or second operator."""
        if np.random.rand() < self.alpha:
            self.perturbate1(X)
        else:
            self.perturbate2(X)

    def perturbate1(self, X: np.ndarray) -> None:
        """
        First perturbation operator
        --> Randomly determines for each game in the schedule independently if the
            game is to be removed with probability self.beta.

        Picked with probability self.alpha.
        """
        drop_mask = (
            (np.random.rand(*X.shape) < self.beta) & np.isfinite(X) & (X < LARGE_NBR)
        )
        X[drop_mask] = np.nan
        np.fill_diagonal(X, LARGE_NBR)

    def perturbate2(self, X: np.ndarray) -> None:
        """
        Second perturbation operator
        --> Chooses a team with a uniform probability, removes all the games of this
            team, and solves the transportation problem for this team.

        Picked with probability 1 - self.alpha.
        """
        idx = np.random.choice(range(X.shape[0]))

        row_schedulable = np.isfinite(X[idx, :]) & (X[idx, :] < LARGE_NBR)
        col_schedulable = np.isfinite(X[:, idx]) & (X[:, idx] < LARGE_NBR)

        X[idx, row_schedulable] = np.nan
        X[col_schedulable, idx] = np.nan
        X[idx, idx] = LARGE_NBR
