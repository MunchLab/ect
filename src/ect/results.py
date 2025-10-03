import matplotlib.pyplot as plt
import numpy as np
from ect.directions import Sampling
from scipy.spatial.distance import cdist
from typing import Union, List, Callable


# ---------- CSR <-> Dense helpers (prefix-difference over thresholds) ----------
def _csr_prefix_to_dense(row_ptr, col_idx, data, num_dirs, num_thresh):
    """Reconstruct dense matrix from CSR of per-row prefix jumps.

    Each row j accumulates jumps at threshold bins given by
    col_idx[row_ptr[j]:row_ptr[j+1]] with magnitudes data[...]. The output is
    the cumulative sum across thresholds [0..num_thresh-1]. Any entries at bin
    num_thresh are interpreted as past the last output index and ignored.
    """
    T = int(num_thresh)
    D = int(num_dirs)
    out = np.zeros((D, T), dtype=np.int64)
    for j in range(D):
        start = int(row_ptr[j])
        end = int(row_ptr[j + 1])
        s = 0
        ptr = start
        for t in range(T):
            while ptr < end and int(col_idx[ptr]) == t:
                s += int(data[ptr])
                ptr += 1
            out[j, t] = s
        # any jumps at bin T are past the last output index by convention
    return out


def _dense_to_csr_prefix(dense64: np.ndarray):
    """Build CSR of per-row prefix jumps from a dense ECT matrix.

    For each row, store non-zero differences of consecutive thresholds:
      jump(0) = dense[0]
      jump(t) = dense[t] - dense[t-1] for t >= 1
    """
    if dense64.ndim != 2:
        raise ValueError("dense matrix must be 2D")
    D, T = dense64.shape
    # collect per-row jumps then trim to actual size
    row_ptr = np.zeros(D + 1, dtype=np.int64)
    jumps_col = []
    jumps_val = []
    nnz = 0
    for j in range(D):
        prev = 0
        for t in range(T):
            cur = int(dense64[j, t])
            delta = cur - prev
            if delta != 0:
                jumps_col.append(t)
                jumps_val.append(delta)
                nnz += 1
            prev = cur
        row_ptr[j + 1] = nnz
    col_idx = np.asarray(jumps_col, dtype=np.int32)
    data = np.asarray(jumps_val, dtype=np.int64)
    return row_ptr, col_idx, data


class ECTResult(np.ndarray):
    """
    A numpy ndarray subclass that carries ECT metadata and plotting capabilities
    Acts like a regular matrix but with added visualization methods and metadata about directions and thresholds
    """

    def __new__(cls, matrix, directions, thresholds):
        # allow float arrays for smooth transform otherwise int
        if np.issubdtype(matrix.dtype, np.floating):
            obj = np.asarray(matrix, dtype=np.float64).view(cls)
        else:
            obj = np.asarray(matrix, dtype=np.int32).view(cls)
        obj.directions = directions
        obj.thresholds = thresholds
        obj.csr_row_ptr = None
        obj.csr_col_idx = None
        obj.csr_data = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.directions = getattr(obj, "directions", None)
        self.thresholds = getattr(obj, "thresholds", None)
        self.csr_row_ptr = getattr(obj, "csr_row_ptr", None)
        self.csr_col_idx = getattr(obj, "csr_col_idx", None)
        self.csr_data = getattr(obj, "csr_data", None)

    @property
    def has_csr(self):
        return (
            getattr(self, "csr_row_ptr", None) is not None
            and getattr(self, "csr_col_idx", None) is not None
            and getattr(self, "csr_data", None) is not None
        )

    @classmethod
    def from_csr(cls, row_ptr, col_idx, data, directions, thresholds, dtype=np.int32):
        num_dirs = len(directions)
        num_thresh = len(thresholds)
        dense64 = _csr_prefix_to_dense(row_ptr, col_idx, data, num_dirs, num_thresh)
        dense = dense64.astype(dtype, copy=False) if dtype == np.int32 else dense64
        obj = cls(dense, directions, thresholds)
        obj.csr_row_ptr = row_ptr
        obj.csr_col_idx = col_idx
        obj.csr_data = data
        return obj

    def to_dense(self):
        if not self.has_csr:
            return self
        num_dirs = self.shape[0]
        num_thresh = self.shape[1]
        dense64 = _csr_prefix_to_dense(
            self.csr_row_ptr, self.csr_col_idx, self.csr_data, num_dirs, num_thresh
        )
        return dense64.astype(self.dtype, copy=False)

    def save_npz(self, path):
        if not self.has_csr:
            row_ptr, col_idx, data = _dense_to_csr_prefix(
                self.astype(np.int64, copy=False)
            )
        else:
            row_ptr, col_idx, data = self.csr_row_ptr, self.csr_col_idx, self.csr_data
        np.savez_compressed(
            path,
            row_ptr=row_ptr,
            col_idx=col_idx,
            data=data,
            thresholds=np.asarray(self.thresholds, dtype=np.float64),
            dtype=str(self.dtype),
        )

    @classmethod
    def load_npz(cls, path, directions):
        z = np.load(path, allow_pickle=False)
        row_ptr = z["row_ptr"]
        col_idx = z["col_idx"]
        data = z["data"]
        thresholds = z["thresholds"]
        dtype = np.dtype(str(z["dtype"]))
        return cls.from_csr(row_ptr, col_idx, data, directions, thresholds, dtype=dtype)

    def plot(self, ax=None):
        """Plot ECT matrix with proper handling for both 2D and 3D"""
        ax = ax or plt.gca()

        if self.thresholds is None:
            self.thresholds = np.linspace(-1, 1, self.shape[1])

        if len(self.directions) == 1:
            directions = (
                self.directions.thetas
                if self.directions.dim == 2
                else self.directions.vectors
            )
            self._plot_ecc(directions)
            return ax

        if self.directions.dim == 2:
            # 2D case - use angle representation
            if (
                self.directions.sampling == Sampling.UNIFORM
                and not self.directions.endpoint
            ):
                plot_thetas = np.concatenate([self.directions.thetas, [2 * np.pi]])
                ect_data = np.hstack([self.T, self.T[:, [0]]])
            else:
                plot_thetas = self.directions.thetas

            X = plot_thetas
            Y = self.thresholds

        else:
            X = np.arange(self.shape[0])
            Y = self.thresholds
            ect_data = self.T

            ax.set_xlabel("Direction Index")

        mesh = ax.pcolormesh(
            X[None, :], Y[:, None], ect_data, cmap="viridis", shading="nearest"
        )
        plt.colorbar(mesh, ax=ax)

        if self.directions.dim == 2:
            ax.set_xlabel(r"Direction $\omega$ (radians)")
            if self.directions.sampling == Sampling.UNIFORM:
                ax.set_xticks(np.linspace(0, 2 * np.pi, 9))
                ax.set_xticklabels(
                    [
                        r"$0$",
                        r"$\frac{\pi}{4}$",
                        r"$\frac{\pi}{2}$",
                        r"$\frac{3\pi}{4}$",
                        r"$\pi$",
                        r"$\frac{5\pi}{4}$",
                        r"$\frac{3\pi}{2}$",
                        r"$\frac{7\pi}{4}$",
                        r"$2\pi$",
                    ]
                )

        ax.set_ylabel(r"Threshold $a$")
        return ax

    def smooth(self):
        """Calculate the Smooth Euler Characteristic Transform"""
        # convert to float for calculations
        data = self.astype(np.float64)

        # get average for each direction
        direction_avgs = np.average(data, axis=1)

        # center each direction's values
        centered = data - direction_avgs[:, np.newaxis]

        # compute cumulative sum to get SECT
        sect = np.cumsum(centered, axis=1)

        # create new ECTResult with float type
        return ECTResult(sect.astype(np.float64), self.directions, self.thresholds)

    def _plot_ecc(self, theta):
        """Plot the Euler Characteristic Curve for a specific direction"""
        plt.step(self.thresholds, self.T, label="ECC")
        theta_round = str(np.round(theta, 2))
        plt.title(r"ECC for $\omega = " + theta_round + "$")
        plt.xlabel("$a$")
        plt.ylabel(r"$\chi(K_a)$")

    def dist(
        self,
        other: Union["ECTResult", List["ECTResult"]],
        metric: Union[str, Callable] = "cityblock",
        **kwargs,
    ):
        """
        Compute distance to another ECTResult or list of ECTResults.

        Args:
            other: Another ECTResult object or list of ECTResult objects
            metric: Distance metric to use. Can be:
                   - String: any metric supported by scipy.spatial.distance
                     (e.g., 'euclidean', 'cityblock', 'chebyshev', 'cosine', etc.)
                   - Callable: a custom distance function that takes two 1D arrays
                     and returns a scalar distance. The function should have signature:
                     func(u, v) -> float
            **kwargs: Additional keyword arguments passed to the metric function
                     (e.g., p=3 for minkowski distance, w=weights for weighted metrics)

        Returns:
            float or np.ndarray: Single distance if other is an ECTResult,
                                 array of distances if other is a list

        Raises:
            ValueError: If the shapes of the ECTResults don't match

        Examples:
            >>> # Built-in metrics
            >>> dist1 = ect1.dist(ect2, metric='euclidean')
            >>> dist2 = ect1.dist(ect2, metric='minkowski', p=3)
            >>>
            >>> # Custom distance function
            >>> def my_distance(u, v):
            ...     return np.sum(np.abs(u - v) ** 0.5)
            >>> dist3 = ect1.dist(ect2, metric=my_distance)
            >>>
            >>> # Batch distances with custom function
            >>> dists = ect1.dist([ect2, ect3, ect4], metric=my_distance)
        """
        # normalize input to list
        single = isinstance(other, ECTResult)
        others = [other] if single else other

        if not others:
            return np.array([])

        for i, ect in enumerate(others):
            if ect.shape != self.shape:
                raise ValueError(
                    f"Shape mismatch at index {i}: {self.shape} vs {ect.shape}"
                )

        # use ravel to avoid copying the data and compute distances
        distances = cdist(
            self.ravel()[np.newaxis, :],
            np.vstack([ect.ravel() for ect in others]),
            metric=metric,
            **kwargs,
        )[0]

        return distances[0] if single else distances
