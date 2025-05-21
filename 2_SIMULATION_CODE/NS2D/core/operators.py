import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

import pywt

class EntropicOperators:
    """
    Classe des opérateurs entropiques et différentiels pour TOEND.
    Supporte CPU/GPU, dimension locale/fractale (n_star), opérateurs classiques et fractals.
    """

    def __init__(self, config, grid, mode="2D"):
        self.config = config
        self.grid = grid
        self.mode = mode
        self.backend = cp if (cp is not None and getattr(config, 'use_gpu', False)) else np
        self.N = config.N
        self.dx = config.L / config.N

        # Grilles spectrales FFT
        k = self.backend.fft.fftfreq(self.N, d=self.dx) * 2 * self.backend.pi
        self.kx, self.ky = self.backend.meshgrid(k, k, indexing="ij")

    def grad(self, f, n_star=None, use_frac=False):
        """
        Calcule le gradient de f.
        Si use_frac=True et n_star fourni, applique une dérivée fractale locale (ondelettes).
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

        # Séparabilité 2D : ondelettes sur chaque axe
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
        """
        Laplacien (avec support pour pondération dimensionnelle via n_star)
        """
        k2 = self.grid["k2"]
        scale = self._local_scaling(n_star) ** 2
        f_hat = self.backend.fft.fft2(f)
        return self.backend.fft.ifft2(-k2 * f_hat * scale).real

    def div(self, fx, fy, n_star=None):
        """
        Divergence d'un champ vectoriel (fx, fy), option dimensionnelle.
        """
        dfx_dx, _ = self.grad(fx, n_star)
        _, dfy_dy = self.grad(fy, n_star)
        return dfx_dx + dfy_dy

    def curl(self, u, v, n_star=None):
        """
        Rotationnel 2D (vorticité), option dimensionnelle.
        """
        _, uy = self.grad(u, n_star)
        vx, _ = self.grad(v, n_star)
        return vx - uy

    def strain_rate(self, u, v, n_star=None):
        """
        Taux de strain 2D (pour la production d'entropie).
        """
        ux, uy = self.grad(u, n_star)
        vx, vy = self.grad(v, n_star)
        return self.backend.sqrt(2*(ux**2 + vy**2) + (uy + vx)**2)

    def lambda_field(self, mu, sigma, n_star=None, energy=None):
        """
        Champ lambda (TOEND) : structure entropique locale.
        Peut intégrer la dimension locale (n_star) et l'énergie (energy).
        """
        dmu_x, dmu_y = self.grad(mu, n_star)
        dsig_x, dsig_y = self.grad(sigma, n_star)
        dot = dmu_x * dsig_x + dmu_y * dsig_y
        norm = dsig_x**2 + dsig_y**2 + 1e-8
        lmbda = dot / norm

        # Régulation dimensionnelle/énergétique optionnelle (TOEND v3)
        if n_star is not None and energy is not None:
            lmbda += 0.1 * n_star + 0.01 * mu * sigma / (energy + 1e-8)
        return lmbda

    def _local_scaling(self, n_star):
        """
        Facteur local (fractal/dimension effect) pour opérateurs fractionnaires.
        """
        if n_star is None:
            return 1.0
        return self.backend.sqrt(n_star / 2)
    def fractal_gradient(self, f, alpha_map, dx=1.0, method="windowed_fourier"):
        """
        Calcul d'une dérivée fractionnaire locale d'ordre alpha(x, y) sur f(x, y).
        Args:
            f: Champ 2D (array)
            alpha_map: Champ 2D (ordre local, entre 0 et 2)
            dx: Pas spatial
            method: 'windowed_fourier' ou 'finite_diff'
        Returns:
            grad_f: Champ dérivé (approximation ∂^α(x,y)f/∂x^α(x,y))
        """
        xp = self.backend
        n = f.shape[0]
        grad_f = xp.zeros_like(f)
        
        if method == "finite_diff":
            # Grünwald-Letnikov local: simple, lent, très local
            for i in range(1, n-1):
                alpha = alpha_map[i]
                # On prend un stencil minimal ici, à généraliser
                grad_f[i] = (f[i+1] - f[i-1]) / (2 * dx ** alpha)
            return grad_f
        
        elif method == "windowed_fourier":
            # FFT fenêtrée: approxime la dérivée locale par fenêtres glissantes
            window_size = 16  # à ajuster selon la résolution
            for i in range(n):
                i0 = max(0, i - window_size // 2)
                i1 = min(n, i + window_size // 2)
                f_local = f[i0:i1]
                alpha = xp.mean(alpha_map[i0:i1])
                k = xp.fft.fftfreq(f_local.size, dx) * 2 * xp.pi
                f_hat = xp.fft.fft(f_local)
                grad_local = xp.fft.ifft((1j * k) ** alpha * f_hat).real
                grad_f[i] = grad_local[window_size // 2] if grad_local.size > window_size // 2 else grad_local[0]
            return grad_f
        else:
            raise ValueError("Unknown method for fractal_gradient")
