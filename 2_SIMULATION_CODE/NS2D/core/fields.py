# core/fields.py
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler

try:
    import cupy as cp
except ImportError:
    cp = None

class FieldContainer:
    """
    Stocke tous les champs physiques et entropiques de la simulation NS2D-TOEND.
    Intègre le calcul de la dimension effective locale (n*).
    """
    def __init__(self, shape, config, use_gpu=False):
        self.use_gpu = use_gpu and cp is not None
        self.xp = cp if self.use_gpu else np
        self.config = config

        # Initialisation des champs
        self.u = self.xp.zeros(shape, dtype=self.xp.float64)      # Vitesse x
        self.v = self.xp.zeros(shape, dtype=self.xp.float64)      # Vitesse y
        self.sigma = self.xp.random.rand(*shape).astype(self.xp.float64) * 0.1
        self.mu = self.xp.zeros(shape, dtype=self.xp.float64)     # Mémoire locale
        self.n_star = self.xp.ones(shape, dtype=self.xp.float64) * 2.0
        self.lmbda = self.xp.zeros(shape, dtype=self.xp.float64)  # Champ λ
        self.S = self.xp.zeros(shape, dtype=self.xp.float64)      # Entropie totale
        self.dirr = self.xp.zeros(shape, dtype=self.xp.float64)   # Dissipation
        self.t = 0.0

        # Processeur de dimension (utilise self.xp)
        self.dim_processor = DimensionField(config, self.xp)

    def evolve_n_star(self, dt):
        """Met à jour n* via l'équation d'évolution."""
        self.n_star = self.dim_processor.evolve_dimension(
            self.n_star, self.mu, self.sigma, (self.u, self.v), dt
        )
        self.n_star = self.xp.clip(self.n_star, self.config.nstar_min, self.config.nstar_max)

    def to_cpu(self):
        """Transfère tous les champs sur le CPU."""
        if self.use_gpu:
            for k in self.__dict__:
                v = getattr(self, k)
                if hasattr(v, 'get'):
                    setattr(self, k, cp.asnumpy(v))
            self.use_gpu = False
            self.xp = np

    def to_gpu(self):
        """Transfère tous les champs sur le GPU (si dispo)."""
        if cp is None:
            raise ImportError("CuPy non installé")
        if not self.use_gpu:
            for k in self.__dict__:
                v = getattr(self, k)
                if isinstance(v, np.ndarray):
                    setattr(self, k, cp.asarray(v))
            self.use_gpu = True
            self.xp = cp
            
    def get_scalar_stats(self):
        """Renvoie les statistiques scalaires."""
        stats = {
            'sigma': float(self.xp.mean(self.sigma)),
            'mu': float(self.xp.mean(self.mu)),
            'S': float(self.xp.mean(self.sigma + self.mu)),
            'lambda_': float(self.xp.mean(self.lmbda)),
            'dirr': float(self.xp.mean(self.dirr)),
            'n_star': float(self.xp.mean(self.n_star)),
            't': float(self.t),
            'sigma_std': float(self.xp.std(self.sigma)),
            'mu_max': float(self.xp.max(self.mu)),
            'n_star_entropy': float(self.xp.mean(-self.n_star * self.xp.log(self.n_star + 1e-8)))
        }
        return stats

class DimensionField:
    """Calcule et fait évoluer la dimension effective locale n*."""
    def __init__(self, config, xp):
        self.config = config
        self.xp = xp  # Module numpy/cupy

    def evolve_dimension(self, n_star, mu, sigma, u, dt):
        """Équation PDE pour n*."""
        D = self.config.D_nstar
        lap = self._laplacian(n_star)
        adv = self._advection(n_star, u) if u is not None else 0.0
        
        # Termes de réaction (avec clipping pour stabilité)
        n_star_clipped = self.xp.clip(n_star, 1e-8, self.config.nstar_max)
        reaction = (
            self.config.alpha_nstar * mu
            - self.config.beta_nstar * sigma * n_star
            + self.config.gamma_nstar * (1 - n_star / self.config.nstar_max)
            + self.config.nu_nstar * n_star_clipped * self.xp.log(n_star_clipped)
        )
        
        # Mise à jour explicite
        n_star_new = n_star + dt * (D * lap - adv + reaction)
        return n_star_new

    def _laplacian(self, field):
        """Laplacien discrétisé (périodique)."""
        return (
            self.xp.roll(field, 1, axis=0)
            + self.xp.roll(field, -1, axis=0)
            + self.xp.roll(field, 1, axis=1)
            + self.xp.roll(field, -1, axis=1)
            - 4 * field
        ) / (self.config.dx ** 2)

    def _advection(self, n_star, u):
        """Advection upwind (plus stable)."""
        u_x, u_y = u
        grad_x = self.xp.where(
            u_x > 0,
            (n_star - self.xp.roll(n_star, 1, axis=0)) / self.config.dx,
            (self.xp.roll(n_star, -1, axis=0) - n_star) / self.config.dx
        )
        grad_y = self.xp.where(
            u_y > 0,
            (n_star - self.xp.roll(n_star, 1, axis=1)) / self.config.dx,
            (self.xp.roll(n_star, -1, axis=1) - n_star) / self.config.dx
        )
        return u_x * grad_x + u_y * grad_y


    # Méthodes optionnelles (ex: compute_fractal_dimension)
    def compute_fractal_dimension(self, field, window_size=8):
        """Dimension fractale via box-counting (exemple simplifié)."""
        # Exemple avec une grille binaire (à adapter)
        threshold = self.xp.mean(field)
        binary_field = (field > threshold).astype(int)
        # Box-counting algorithm (version simplifiée)
        scales = self.xp.logspace(0, 3, num=10, base=2)
        counts = []
        for scale in scales:
            scaled = binary_field[::int(scale), ::int(scale)]
            counts.append(self.xp.sum(scaled))
        slope = self.xp.polyfit(self.xp.log(scales), self.xp.log(counts), 1)[0]
        return -slope

    def compute_spectral_dimension(self, laplacian):
        """Dimension spectrale via décroissance des valeurs propres."""
        eigenvalues = self.xp.linalg.eigvalsh(laplacian)
        sorted_eig = self.xp.sort(eigenvalues)[::-1]
        decay_rate = self.xp.polyfit(self.xp.arange(len(sorted_eig)), self.xp.log(sorted_eig), 1)[0]
        return decay_rate * 2  # Ajustement empirique

    def compute_information_dimension(self, density):
        """Dimension informationnelle via entropie de Shannon."""
        prob = density / self.xp.sum(density)
        entropy = -self.xp.sum(prob * self.xp.log(prob + 1e-12))
        return entropy / self.xp.log(self.xp.prod(density.shape))
