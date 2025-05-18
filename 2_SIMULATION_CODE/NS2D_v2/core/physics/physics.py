import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from scipy.ndimage import gaussian_filter

class PhysicsCore:
    def __init__(self, config, operators):
        self.config = config
        self.ops = operators
        self.debug_data = [] if config.debug_mode else None

    def update_n_star(self, u, v, sigma):
        omega = self.ops.curl(u, v)
        omega_x, omega_y = self.ops.grad(omega)
        sigma_x, _ = self.ops.grad(sigma)

        omega_term = np.sqrt(omega_x**2 + omega_y**2)
        sigma_term = np.abs(sigma_x)

        raw_n = 2.3 + 0.5 * (omega_term + 0.3 * sigma_term - 2.3)
        smoothed_n = gaussian_filter(raw_n, sigma=1.0)
        return np.clip(smoothed_n, *self.config.nstar_bounds)

    def entropy_production(self, S, sigma):
        S = np.clip(S, 1e-6, self.config.strain_max)
        sigma = np.clip(sigma, 1e-8, self.config.sigma_max)
        return self.config.nu * S**2 * (1 + sigma * S**self.config.beta)

    def memory_feedback(self, mu, S):
        S_clip = np.clip(S, 0, self.config.strain_max)
        mu_clip = np.clip(mu, 0, self.config.mu_max)
        return self.config.gamma * mu_clip * np.tanh(S_clip**2 / 0.3)
