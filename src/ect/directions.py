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

    **Examples:**
    
    .. code-block:: python

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
        """
        Initialize a Directions instance for ECT calculations.

        Args:
            num_dirs (int): Number of direction vectors to generate (default: 360).
            sampling (Sampling): Sampling strategy (UNIFORM, RANDOM, or CUSTOM).
            dim (int): Dimension of the space (default: 2).
            endpoint (bool): Whether to include the endpoint for 2D angles (default: False).
            seed (Optional[int]): Random seed for reproducibility (default: None).

        Notes:
            - For 2D, directions are represented as angles; for higher dimensions, as unit vectors.
            - Use factory methods :meth:`uniform`, :meth:`random`, :meth:`from_angles`, or :meth:`from_vectors` for convenience.
        """
        
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
        Create a Directions instance for custom angles in 2D.

        Args:
            angles (Sequence[float]): List or array of angles (in radians) for each direction.

        Returns:
            Directions: Instance with direction angles set and corresponding unit vectors available via :attr:`vectors`.

        Notes:
            - Only valid for 2D directions; for higher dimensions use :meth:`from_vectors`.
            - Angles are stored in :attr:`thetas` and unit vectors are computed as needed.
        """
        instance = cls(len(angles), Sampling.CUSTOM, dim=2)
        instance._thetas = np.array(angles)
        return instance

    @classmethod
    def from_vectors(cls, vectors: Sequence[tuple]) -> "Directions":
        """
        Create a Directions instance from custom direction vectors in any dimension.

        Args:
            vectors (Sequence[tuple]): List or array of direction vectors (each must be nonzero).

        Returns:
            Directions: Instance with normalized direction vectors and associated angles (if 2D).

        Raises:
            ValueError: If any vector has zero magnitude.

        Notes:
            - Vectors are normalized to unit length.
            - For 2D, angles are computed from the vectors and available via :attr:`thetas`.
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
        """
        Get the angles (in radians) for 2D direction vectors.

        Returns:
            np.ndarray: Array of angles corresponding to each direction vector (2D only).

        Raises:
            ValueError: If called for directions in dimension greater than 2.

        Notes:
            - For 2D, angles are computed from the direction vectors if not already set.
            - For higher dimensions, use :attr:`vectors` for direction data.
        """
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
        """
        Get the unit direction vectors for all directions.

        Returns:
            np.ndarray: Array of shape (num_dirs, dim) containing unit vectors for each direction.

        Raises:
            ValueError: If vectors are not available for dimensions >2 and were not generated during initialization.

        Notes:
            - For 2D, vectors are computed from :attr:`thetas` if not already set.
            - For higher dimensions, vectors are generated during initialization or via :meth:`from_vectors`.
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
