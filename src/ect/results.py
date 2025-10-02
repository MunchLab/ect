import matplotlib.pyplot as plt
import numpy as np
from ect.directions import Sampling
from scipy.spatial.distance import cdist
from typing import Union, List, Callable


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
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.directions = getattr(obj, "directions", None)
        self.thresholds = getattr(obj, "thresholds", None)

    def plot(self, ax=None, *, radial=False, **kwargs):
        """Plot ECT matrix with proper handling for both 2D and 3D.

        Set radial=True to render a polar visualization (2D only). Any extra
        keyword arguments are forwarded to the radial renderer.
        """
        if radial:
            return self._plot_radial(ax=ax, **kwargs)

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

    # Internal plotting utilities
    def _ensure_2d(self):
        if self.directions is None or self.directions.dim != 2:
            raise ValueError("This visualization is only supported for 2D ECT results")

    def _theta_threshold_mesh(self):
        thetas = self.directions.thetas
        thresholds = self.thresholds
        THETA, R = np.meshgrid(thetas, thresholds)
        return THETA, R

    def _configure_polar_axes(
        self, ax, rmin=0.0, rmax=None, theta_zero="N", theta_dir=-1
    ):
        ax.set_theta_zero_location(theta_zero)
        ax.set_theta_direction(theta_dir)
        if rmax is None:
            rmax = float(np.max(self.thresholds))
        ax.set_ylim(float(rmin), float(rmax))
        return ax

    def _scale_overlay_radii(self, points, rmin=0.0, rmax=None, fit_to_thresholds=True):
        x = points[:, 0]
        y = points[:, 1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        if rmax is None:
            rmax = float(np.max(self.thresholds))

        if not fit_to_thresholds:
            return theta, r

        max_r_points = float(np.max(r)) if r.size else 0.0
        if max_r_points > 0.0:
            scaled_r = (r / max_r_points) * (rmax - float(rmin)) + float(rmin)
        else:
            scaled_r = r
        return theta, scaled_r

    def _plot_ecc(self, theta):
        """Plot the Euler Characteristic Curve for a specific direction"""
        plt.step(self.thresholds, self.T, label="ECC")
        theta_round = str(np.round(theta, 2))
        plt.title(r"ECC for $\omega = " + theta_round + "$")
        plt.xlabel("$a$")
        plt.ylabel(r"$\chi(K_a)$")

    def _plot_radial(
        self,
        ax=None,
        title=None,
        cmap="viridis",
        *,
        rmin=0.0,
        rmax=None,
        colorbar=True,
        overlay=None,
        overlay_kwargs=None,
        **kwargs,
    ):
        """
        Plot ECT matrix in polar coordinates (radial plot).

        Args:
            ax: matplotlib axes object. If None, creates a new polar subplot
            title: optional string for plot title
            cmap: colormap for the plot (default: 'viridis')
            rmin: minimum radius for the plot (default: 0.0)
            rmax: maximum radius for the plot (default: None)
            colorbar: whether to show the colorbar (default: True)
            overlay: points to overlay on the plot (default: None)

            **kwargs: additional keyword arguments passed to pcolormesh

        Returns:
            matplotlib.axes.Axes: The axes object used for plotting
        """
        self._ensure_2d()

        if ax is None:
            fig, ax = plt.subplots(
                subplot_kw=dict(projection="polar"), figsize=(10, 10)
            )

        THETA, R = self._theta_threshold_mesh()

        im = ax.pcolormesh(THETA, R, self.T, cmap=cmap, **kwargs)

        self._configure_polar_axes(ax, rmin=rmin, rmax=rmax)

        if title:
            ax.set_title(title)

        if colorbar:
            plt.colorbar(im, ax=ax, label="ECT Value")

        if overlay is not None:
            overlay_kwargs = overlay_kwargs or {}
            theta, scaled_r = self._scale_overlay_radii(
                overlay, rmin=rmin, rmax=rmax, fit_to_thresholds=True
            )
            ax.plot(
                theta,
                scaled_r,
                "-",
                color=overlay_kwargs.get("color", "black"),
                linewidth=overlay_kwargs.get("linewidth", 2),
                alpha=overlay_kwargs.get("alpha", 0.5),
            )

        return ax

    def _overlay_points(
        self,
        points,
        ax=None,
        color="black",
        linewidth=2,
        alpha=0.5,
        *,
        rmin=0.0,
        rmax=None,
        fit_to_thresholds=True,
        **kwargs,
    ):
        """
        Overlay original points on a radial ECT plot.

        Args:
            points: numpy array of shape (N, 2) containing the original points
            ax: matplotlib polar axes object. If None, uses current axes
            color: color for the overlay line (default: 'white')
            linewidth: line width for the overlay (default: 2)
            alpha: transparency for the overlay (default: 0.5)
            **kwargs: additional keyword arguments passed to plot

        Returns:
            matplotlib.axes.Axes: The axes object used for plotting
        """
        if ax is None:
            ax = plt.gca()

        if not hasattr(ax, "name") or ax.name != "polar":
            raise ValueError("overlay_points requires a polar axes object")

        theta, scaled_r = self._scale_overlay_radii(
            points, rmin=rmin, rmax=rmax, fit_to_thresholds=fit_to_thresholds
        )

        ax.plot(
            theta,
            scaled_r,
            "-",
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            **kwargs,
        )

        return ax

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
