# core/fields.py
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

class FieldContainer:
    """
    Stocke tous les champs physiques et entropiques de la simulation NS2D-TOEND.
    Supporte CPU/GPU et conversions.
    """
    def __init__(self, shape, use_gpu=False):
        self.use_gpu = use_gpu and cp is not None
        self.xp = cp if self.use_gpu else np

        self.u = self.xp.zeros(shape, dtype=self.xp.float64)      # Vitesse x
        self.v = self.xp.zeros(shape, dtype=self.xp.float64)      # Vitesse y
        self.sigma = self.xp.random.rand(*shape).astype(self.xp.float64) * 0.1   # Entropie locale (σ)
        self.mu = self.xp.zeros(shape, dtype=self.xp.float64)     # Mémoire locale (μ)
        self.n_star = self.xp.ones(shape, dtype=self.xp.float64) * 2.0           # Dimension effective locale (n*)
        self.lmbda = self.xp.zeros(shape, dtype=self.xp.float64)  # Champ λ(x, y, t) (optionnel)
        self.S = self.xp.zeros(shape, dtype=self.xp.float64)      # Entropie totale locale (σ + μ)
        self.dirr = self.xp.zeros(shape, dtype=self.xp.float64)   # Dissipation informationnelle
        self.t = 0.0

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
        """
        Renvoie un dict avec les proxies scalaires (moyennes spatiales) :
        σ, μ, S, λ, dirr (pour diagnostics et plot)
        """
        xp = self.xp
        return dict(
            sigma=float(xp.mean(self.sigma)),
            mu=float(xp.mean(self.mu)),
            S=float(xp.mean(self.sigma + self.mu)),
            lambda_=float(xp.mean(self.lmbda)),
            dirr=float(xp.mean(self.dirr)),
            n_star=float(xp.mean(self.n_star)),
            t=float(self.t)
        )
