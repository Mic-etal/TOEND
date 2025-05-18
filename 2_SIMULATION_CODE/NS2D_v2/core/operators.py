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

class EntropicNS2DOperators:
    def __init__(self, config, grid):
        self.config = config
        self.grid = grid
        self.backend = cp if config.use_gpu else np
        self.fft2 = cp.fft.fft2 if config.use_gpu else np.fft.fft2
        self.ifft2 = cp.fft.ifft2 if config.use_gpu else np.fft.ifft2

    def grad(self, f, n_star=None):
        """Spectral gradient with optional n* modulation"""
        scale = np.sqrt(n_star/2) if n_star is not None else 1.0
        f_hat = self.fft2(f)
        fx = self.ifft2(1j * self.grid['kx'] * scale * f_hat).real
        fy = self.ifft2(1j * self.grid['ky'] * scale * f_hat).real
        return fx, fy

    def laplace(self, f, n_star=None):
        """Spectral Laplacian with optional n* modulation"""
        k2 = (self.grid['kx']**2 + self.grid['ky']**2)
        if n_star is not None:
            k2 = k2 * (n_star / 2)
        return self.ifft2(-k2 * self.fft2(f)).real

    def div(self, fx, fy, n_star=None):
        """Divergence operator with n* awareness"""
        dfx_dx, _ = self.grad(fx, n_star)
        _, dfy_dy = self.grad(fy, n_star)
        return dfx_dx + dfy_dy

    def curl(self, u, v, n_star=None):
        """Vorticity calculation"""
        _, uy = self.grad(u, n_star)
        vx, _ = self.grad(v, n_star)
        return vx - uy

    def strain_rate(self, u, v, n_star=None):
        """Compute strain rate tensor magnitude"""
        ux, uy = self.grad(u, n_star)
        vx, vy = self.grad(v, n_star)
        return np.sqrt(2*(ux**2 + vy**2) + (uy + vx)**2)