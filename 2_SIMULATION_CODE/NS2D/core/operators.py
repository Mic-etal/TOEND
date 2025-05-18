import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

import pywt

class EntropicOperators:
    """
    Classe des opérateurs entropiques avec support CPU/GPU.
    Inclut dérivées classiques et fractales locales par ondelettes.
    """
    def __init__(self, config, grid, mode="2D"):
        self.config = config
        self.grid = grid
        self.mode = mode
        self.backend = cp if (cp is not None and getattr(config, 'use_gpu', False)) else np
        self.N = config.N
        self.dx = config.L / config.N
        k = self.backend.fft.fftfreq(self.N, d=self.dx) * 2 * self.backend.pi
        self.kx, self.ky = self.backend.meshgrid(k, k, indexing="ij")

    def grad(self, f, n_star=None, use_frac=False):
        """
        Calcule le gradient de f.
        Si use_frac est True et n_star fourni, applique une dérivée fractale locale.
        """
        if use_frac:
            if n_star is None:
                raise ValueError("Pour use_frac=True, n_star doit être fourni.")
            return self._fract_local_gradient(f, n_star)

        f_hat = self.backend.fft.fft2(f)
        scale = self._local_scaling(n_star)
        fx = self.backend.fft.ifft2(1j * self.kx * f_hat * scale).real
        fy = self.backend.fft.ifft2(1j * self.ky * f_hat * scale).real
        return fx, fy

    def _fract_local_gradient(self, f, alpha_map, wavelet='db4'):
        """
        Dérivée fractale locale approximée par ondelettes discrètes pondérées.

        Args:
            f (array): champ scalaire
            alpha_map (array): champ local d'ordre fractal
            wavelet (str): type d'ondelette à utiliser

        Returns:
            tuple: (df/dx, df/dy) approximé fractalement
        """
        xp = self.backend

        # Conversion CPU/GPU
        if isinstance(f, np.ndarray) and xp == cp:
            f = cp.asarray(f)
        elif isinstance(f, cp.ndarray) and xp == np:
            f = cp.asnumpy(f)

        # Pour l'instant, on applique la dérivée fractale de façon isotrope en 2D par séparabilité
        coeffs_x = pywt.wavedec(f.get() if xp == cp else f, wavelet, mode='periodization', axis=0)
        coeffs_y = pywt.wavedec(f.get() if xp == cp else f, wavelet, mode='periodization', axis=1)

        alpha_mean = xp.mean(alpha_map)

        # Pondération indicative des coefficients (prototype)
        for i in range(len(coeffs_x)):
            coeffs_x[i] = coeffs_x[i] * (1j ** alpha_mean)
        for i in range(len(coeffs_y)):
            coeffs_y[i] = coeffs_y[i] * (1j ** alpha_mean)

        df_dx = pywt.waverec(coeffs_x, wavelet, mode='periodization', axis=0)
        df_dy = pywt.waverec(coeffs_y, wavelet, mode='periodization', axis=1)

        return xp.asarray(df_dx), xp.asarray(df_dy)

    def laplace(self, f, n_star=None):
        k2 = self.grid["k2"]
        scale = self._local_scaling(n_star)**2
        f_hat = self.backend.fft.fft2(f)
        return self.backend.fft.ifft2(-k2 * f_hat * scale).real

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
        return self.backend.sqrt(2*(ux**2 + vy**2) + (uy + vx)**2)

    def lambda_field(self, mu, sigma):
        dmu_x, dmu_y = self.grad(mu)
        dsig_x, dsig_y = self.grad(sigma)
        dot = dmu_x * dsig_x + dmu_y * dsig_y
        norm = dsig_x**2 + dsig_y**2 + 1e-8
        return dot / norm

    def _local_scaling(self, n_star):
        if n_star is None:
            return 1.0
        xp = self.backend
        return xp.sqrt(n_star / 2)
