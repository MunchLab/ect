import numpy as np
from typing import Union, Optional, List, Sequence
from enum import Enum


class Sampling(Enum):
    UNIFORM = "uniform"
    RANDOM = "random"
    CUSTOM = "custom"


class Directions:
    """
    Manages direction vectors for ECT calculations.
    Supports uniform, random, or custom sampling of directions.

    Example:
        # Uniform sampling
        dirs = Directions(num_dirs=8)

        # Random sampling
        dirs = Directions.random(num_dirs=10, seed=42)

        # Custom angles
        dirs = Directions.from_angles([0, np.pi/4, np.pi/2])

        # Custom vectors
        dirs = Directions.from_vectors([(1,0), (1,1), (0,1)])
    """

    def __init__(self,
                 num_dirs: int = 360,
                 sampling: Sampling = Sampling.UNIFORM,
                 endpoint: bool = False,
                 seed: Optional[int] = None):

        self.num_dirs = num_dirs
        self.sampling = sampling
        self.endpoint = endpoint

        if seed is not None:
            np.random.seed(seed)

        self._thetas = None
        self._vectors = None
        self._initialize_directions()

    def _initialize_directions(self):
        """Initialize direction angles based on strategy"""
        if self.sampling == Sampling.UNIFORM:
            self._thetas = np.linspace(0, 2*np.pi,
                                       self.num_dirs,
                                       endpoint=self.endpoint)
        elif self.sampling == Sampling.RANDOM:
            self._thetas = np.random.uniform(0, 2*np.pi, self.num_dirs)
            self._thetas.sort()  # sort for consistency

    @classmethod
    def random(cls, num_dirs: int = 360, seed: Optional[int] = None) -> 'Directions':
        """Create instance with random direction sampling"""
        return cls(num_dirs, Sampling.RANDOM, seed=seed)

    @classmethod
    def from_angles(cls, angles: Sequence[float]) -> 'Directions':
        """Create instance from custom angles"""
        instance = cls(len(angles), Sampling.CUSTOM)
        instance._thetas = np.array(angles)
        return instance

    @classmethod
    def from_vectors(cls, vectors: Sequence[tuple]) -> 'Directions':
        """Create instance from custom direction vectors"""
        vectors = np.array(vectors)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / norms

        instance = cls(len(vectors), Sampling.CUSTOM)
        instance._vectors = normalized
        instance._thetas = np.arctan2(normalized[:, 1], normalized[:, 0])
        return instance

    @property
    def thetas(self) -> np.ndarray:
        """Get angles for all directions"""
        return self._thetas

    @property
    def vectors(self) -> np.ndarray:
        """Get unit vectors for all directions"""
        if self._vectors is None:
            self._vectors = np.column_stack([
                np.cos(self._thetas),
                np.sin(self._thetas)
            ])
        return self._vectors

    def __len__(self) -> int:
        return self.num_dirs

    def __getitem__(self, idx) -> np.ndarray:
        """Get direction vector at index"""
        return self.vectors[idx]
