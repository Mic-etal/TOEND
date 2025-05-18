import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from scipy.ndimage import gaussian_filter
import os
# config.py
from dataclasses import dataclass
# operators.py
import numpy as np
from scipy.ndimage import gaussian_filter

class EntropicOperators:
    def __init__(self, config, grid, mode="2D"):
        self.config = config
        self.grid = grid
        self.mode = mode
        self.backend = np  # cupy possible

    def grad(self, f, n_star=None):
        kx, ky = self.grid["kx"], self.grid["ky"]
        f_hat = np.fft.fft2(f)
        scale = self._local_scaling(n_star)
        fx = np.fft.ifft2(1j * kx * f_hat * scale).real
        fy = np.fft.ifft2(1j * ky * f_hat * scale).real
        return fx, fy

    def laplace(self, f, n_star=None):
        k2 = self.grid["k2"]
        scale = self._local_scaling(n_star)**2
        f_hat = np.fft.fft2(f)
        return np.fft.ifft2(-k2 * f_hat * scale).real

    def div(self, fx, fy, n_star=None):
        dfx_dx, _ = self.grad(fx, n_star)
        _, dfy_dy = self.grad(fy, n_star)
        return dfx_dx + dfy_dy

    def curl(self, u, v, n_star=None):
        _, uy = self.grad(u, n_star)
        vx, _ = self.grad(v, n_star)
        return vx - uy

    def strain_rate(self, u, v, n_star=None):
        ux, uy = self.grad(u, n_star)
        vx, vy = self.grad(v, n_star)
        return np.sqrt(2*(ux**2 + vy**2) + (uy + vx)**2)

    def lambda_field(self, mu, sigma):
        dmu_x, dmu_y = self.grad(mu)
        dsig_x, dsig_y = self.grad(sigma)
        dot = dmu_x * dsig_x + dmu_y * dsig_y
        norm = dsig_x**2 + dsig_y**2 + 1e-8
        return dot / norm

    def _local_scaling(self, n_star):
        if n_star is None:
            return 1.0
        return np.sqrt(n_star / 2)
