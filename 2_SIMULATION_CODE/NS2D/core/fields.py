# core/fields.py
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

class FieldContainer:
    def __init__(self, shape, use_gpu=False):
        self.use_gpu = use_gpu and cp is not None
        self.xp = cp if self.use_gpu else np

        self.u = self.xp.zeros(shape, dtype=self.xp.float64)
        self.v = self.xp.zeros(shape, dtype=self.xp.float64)
        self.sigma = self.xp.random.rand(*shape).astype(self.xp.float64) * 0.1
        self.mu = self.xp.zeros(shape, dtype=self.xp.float64)
        self.n_star = self.xp.ones(shape, dtype=self.xp.float64) * 2.0
        self.t = 0.0

    def to_cpu(self):
        if self.use_gpu:
            self.u = cp.asnumpy(self.u)
            self.v = cp.asnumpy(self.v)
            self.sigma = cp.asnumpy(self.sigma)
            self.mu = cp.asnumpy(self.mu)
            self.n_star = cp.asnumpy(self.n_star)
            self.use_gpu = False
            self.xp = np

    def to_gpu(self):
        if cp is None:
            raise ImportError("CuPy not installed")
        if not self.use_gpu:
            self.u = cp.asarray(self.u)
            self.v = cp.asarray(self.v)
            self.sigma = cp.asarray(self.sigma)
            self.mu = cp.asarray(self.mu)
            self.n_star = cp.asarray(self.n_star)
            self.use_gpu = True
            self.xp = cp
