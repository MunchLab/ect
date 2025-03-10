import matplotlib.pyplot as plt
import numpy as np
from ect.directions import Sampling


class ECTResult(np.ndarray):
    """
    A numpy ndarray subclass that carries ECT metadata and plotting capabilities
    Acts like a regular matrix but with added visualization methods
    """
    def __new__(cls, matrix, directions, thresholds):
        obj = np.asarray(matrix).view(cls)
        obj.directions = directions
        obj.thresholds = thresholds
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.directions = getattr(obj, 'directions', None)
        self.thresholds = getattr(obj, 'thresholds', None)

    def plot(self, ax=None):
        """Plot ECT matrix with proper handling for both 2D and 3D"""
        ax = ax or plt.gca()

        if self.thresholds is None:
            self.thresholds = np.linspace(-1, 1, self.shape[1])

        if self.directions.dim == 2:
            # 2D case - use angle representation
            if self.directions.sampling == Sampling.UNIFORM and not self.directions.endpoint:
                plot_thetas = np.concatenate(
                    [self.directions.thetas, [2*np.pi]])
                # Circular closure
                ect_data = np.hstack([self.T, self.T[:, [0]]])
            else:
                plot_thetas = self.directions.thetas
                ect_data = self.T

            X = plot_thetas
            Y = self.thresholds

        else:
            X = np.arange(self.shape[0])
            Y = self.thresholds
            ect_data = self.T

            ax.set_xlabel('Direction Index')

        mesh = ax.pcolormesh(X[None, :], Y[:, None], ect_data,
                             cmap='viridis', shading='nearest')
        plt.colorbar(mesh, ax=ax)

        if self.directions.dim == 2:
            ax.set_xlabel(r'Direction $\omega$ (radians)')
            if self.directions.sampling == Sampling.UNIFORM:
                ax.set_xticks(np.linspace(0, 2*np.pi, 9))
                ax.set_xticklabels([
                    r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',
                    r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{5\pi}{4}$',
                    r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$', r'$2\pi$'
                ])

        ax.set_ylabel(r'Threshold $a$')
        return ax
    

    def smooth(self):
        """
        Function to calculate the Smooth Euler Characteristic Transform (SECT) from the ECT matrix. 

        Returns:
            ECTResult: The SECT matrix with same directions and thresholds
        """
        avg = np.average(self, axis=1)

        #subtract the average from each row
        centered_ect = self - avg[:, np.newaxis]
        
        # take the cumulative sum of each row to get the SECT
        sect = np.cumsum(centered_ect, axis=1)
        
        # Return as ECTResult to maintain plotting capability
        return ECTResult(sect, self.directions, self.thresholds)
