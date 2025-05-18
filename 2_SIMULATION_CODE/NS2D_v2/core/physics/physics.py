import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from scipy.ndimage import gaussian_filter
import time  # Ajout pour le debug

class PhysicsCore:
    def __init__(self, config, operators):
        self.config = config
        self.ops = operators
        self.debug_data = [] if config.debug_mode else None

    def update_n_star(self, u, v, sigma):
        """Adaptive local dimensionality field"""
        omega = self.ops.curl(u, v)
        omega_x, omega_y = self.ops.grad(omega)
        sigma_x, _ = self.ops.grad(sigma)
        
        omega_term = np.sqrt(omega_x**2 + omega_y**2)
        sigma_term = np.abs(sigma_x)
        
        raw_n = 2.3 + 0.5*(omega_term + 0.3*sigma_term - 2.3)
        smoothed_n = gaussian_filter(raw_n, sigma=1.0)
        return np.clip(smoothed_n, *self.config.nstar_bounds)

    def entropy_production(self, S, sigma):
        """Entropy source term with clipping"""
        S = np.clip(S, 1e-6, self.config.strain_max)
        sigma = np.clip(sigma, 1e-8, self.config.sigma_max)
        return self.config.nu * S**2 * (1 + sigma * S**self.config.beta)

    def memory_feedback(self, mu, S):
        """Feedback term with saturation and clipping"""
        S_clip = np.clip(S, 0, self.config.strain_max)
        mu_clip = np.clip(mu, 0, self.config.mu_max)
        return self.config.gamma * mu_clip * np.tanh(S_clip**2 / 0.3)
        
def mu_rhs(sigma, mu, gamma, mu_max):
    return gamma * sigma * (1 - mu / mu_max)

        
def sigma_rhs(sigma, mu, dx, alpha, beta, eta, lam, theta):
    grad_sigma = np.gradient(sigma, dx)
    S = np.abs(grad_sigma) + 1e-6
    
    # Adaptative diffusion
    eta_eff = eta / (1 + np.maximum(mu, 1e-4))
    diffusion = eta_eff * np.gradient(np.gradient(sigma, dx), dx)
    
    # Injection modifiée par la mémoire
    injection = alpha * np.sqrt(np.maximum(sigma, 1e-8)) * S**1.5 * (1 + theta * mu)
    
    # Décroissance non linéaire
    decay = -lam * sigma * S**(2 - beta)

    return diffusion + injection + decay

        
        if self.config.debug_mode:
            self.debug_data.append({
                't': time.time(),
                'strain': np.max(S_clip),
                'sigma': np.mean(sigma_clip),
                'injection': np.mean(injection),
                'decay': np.mean(decay)
            })
        
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)