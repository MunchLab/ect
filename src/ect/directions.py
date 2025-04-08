from typing import Optional, Sequence
from enum import Enum

import numpy as np


class Sampling(Enum):
    UNIFORM = "uniform"
    RANDOM = "random"
    CUSTOM = "custom"


class Directions:
    """
    Manages direction vectors for ECT calculations.
    Supports uniform, random, or custom sampling of directions.

    Examples:
        # Uniform sampling in 2D (default)
        dirs = Directions.uniform(num_dirs=8)

        # Uniform sampling in 3D
        dirs = Directions.uniform(num_dirs=10, dim=3)

        # Random sampling in 2D
        dirs = Directions.random(num_dirs=10, seed=42)

        # Custom angles (2D only)
        dirs = Directions.from_angles([0, np.pi/4, np.pi/2])

        # Custom vectors in any dimension
        dirs = Directions.from_vectors([(1,0,0), (0,1,0), (0,0,1)])
    """

    def __init__(
        self,
        num_dirs: int = 360,
        sampling: Sampling = Sampling.UNIFORM,
        dim: int = 2,
        endpoint: bool = False,
        seed: Optional[int] = None,
    ):
        self.num_dirs = num_dirs
        self.sampling = sampling
        self.dim = dim
        self.endpoint = endpoint

        self._rng = np.random.RandomState(seed)
        self._thetas = None
        self._vectors = None
        self._initialize_directions()

    def _initialize_directions(self):
        """
        Initialize direction vectors using the chosen sampling strategy.
        For 2D, the angles may be stored; for n-dim (n>2) the vectors are generated from
        random normal samples and normalized to lie on the unit sphere.
        """
        if self.sampling == Sampling.UNIFORM:
            if self.dim == 2:
                self._thetas = np.linspace(
                    0, 2 * np.pi, self.num_dirs, endpoint=self.endpoint
                )
            else:
                # generate random normal samples and normalize to lie on the unit sphere
                self._vectors = self._rng.randn(self.num_dirs, self.dim)
                self._vectors /= np.linalg.norm(self._vectors,
                                                axis=1, keepdims=True)
        elif self.sampling == Sampling.RANDOM:
            if self.dim == 2:
                self._thetas = self._rng.uniform(0, 2 * np.pi, self.num_dirs)
                self._thetas.sort()
            else:
                self._vectors = self._rng.randn(self.num_dirs, self.dim)
                self._vectors /= np.linalg.norm(self._vectors,
                                                axis=1, keepdims=True)

    @classmethod
    def uniform(
        cls,
        num_dirs: int = 360,
        dim: int = 2,
        endpoint: bool = False,
        seed: Optional[int] = None,
    ) -> "Directions":
        """
        Factory method for uniform sampling.

        Parameters:
            num_dirs: Number of direction vectors.
            dim: Dimension of the space (default 2).
            endpoint: Whether to include the endpoint (for 2D angles).
            seed: Optional random seed.
        """
        return cls(num_dirs, Sampling.UNIFORM, dim, endpoint, seed)

    @classmethod
    def random(
        cls, num_dirs: int = 360, dim: int = 2, seed: Optional[int] = None
    ) -> "Directions":
        """
        Factory method for random sampling.

        Parameters:
            num_dirs: Number of direction vectors.
            dim: Dimension of the space.
            seed: optional random seed.
        """
        return cls(num_dirs, Sampling.RANDOM, dim, seed=seed)

    @classmethod
    def from_angles(cls, angles: Sequence[float]) -> "Directions":
        """
        Create an instance for custom angles (2D only).
        """
        instance = cls(len(angles), Sampling.CUSTOM, dim=2)
        instance._thetas = np.array(angles)
        return instance

    @classmethod
    def from_vectors(cls, vectors: Sequence[tuple]) -> "Directions":
        """
        Create an instance from custom direction vectors.
        Works in any number of dimensions.
        """
        vectors = np.array(vectors, dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zero-magnitude vectors are not allowed")
        normalized = vectors / norms
        instance = cls(len(vectors), Sampling.CUSTOM, dim=vectors.shape[1])
        instance._vectors = normalized
        if instance.dim == 2:
            instance._thetas = np.arctan2(normalized[:, 1], normalized[:, 0])
        return instance

    @property
    def thetas(self) -> np.ndarray:
        """Get the angles for 2D directions. Raises an error if dim > 2."""
        if self.dim != 2:
            raise ValueError(
                "Angle representation is only available for 2D directions."
            )
        if self._thetas is None:
            # Compute the angles from the vectors.
            self._thetas = np.arctan2(self.vectors[:, 1], self.vectors[:, 0])
        return self._thetas

    @property
    def vectors(self) -> np.ndarray:
        """Get unit direction vectors.
        For 2D, they are computed from thetas if not already created.
        For n-dim (n>2), they should be available.
        """
        if self._vectors is None:
            if self.dim == 2:
                self._vectors = np.column_stack(
                    (np.cos(self._thetas), np.sin(self._thetas))
                )
            else:
                raise ValueError(
                    "Direction vectors for dimensions >2 should be generated during initialization."
                )
        return self._vectors

    def __len__(self) -> int:
        return self.num_dirs

    def __getitem__(self, idx) -> np.ndarray:
        """Return the direction vector at index idx."""
        return self.vectors[idx]
