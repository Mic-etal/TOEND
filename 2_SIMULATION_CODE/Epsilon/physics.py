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
        
    def mu_fusion(self, mu1, mu2, gamma12):
    """Non-associative fusion: μ1 ⊗ μ2 = μ1 + μ2 + γ12 μ1 μ2"""
    return mu1 + mu2 + gamma12 * mu1 * mu2

    def compute_gamma12(self, sigma1, sigma2, L1, L2):
    """Context-dependent coupling coefficient"""
    # Example: γ12 depends on entropy and scale
    return self.config.gamma0 * (sigma1 * sigma2) / (L1 * L2)
        
    def sigma_rhs(self, sigma, mu, grad_sigma):
    """TOEND-compliant entropy dynamics"""
    S = np.abs(grad_sigma) + 1e-6
    diffusion = self.config.eta * np.gradient(np.gradient(sigma, self.dx), self.dx)
    injection = self.config.alpha * np.sqrt(sigma) * S**1.5
    decay = -self.config.lam * sigma * S**(2 - self.config.beta)
    # Add μ feedback
    decay *= (1 - mu / self.config.mu_max)
    return diffusion + injection + decay

    def mu_rhs(self, sigma, mu, grad_sigma):
    """Memory dynamics with non-linear saturation"""
    S = np.abs(grad_sigma) + 1e-6
    growth = self.config.gamma * sigma * S**0.5
    saturation = -self.config.gamma_decay * mu**2
    return growth + saturation
    
    # Adaptative diffusion
    eta_eff = eta / (1 + np.maximum(mu, 1e-4))
    diffusion = eta_eff * np.gradient(np.gradient(sigma, dx), dx)
    
    # Injection modifiée par la mémoire
    injection = alpha * np.sqrt(np.maximum(sigma, 1e-8)) * S**1.5 * (1 + theta * mu)
    
    # Décroissance non linéaire
    decay = -lam * sigma * S**(2 - beta)
    
    def entropy_production(self, S, sigma):
    S_bits = S / (self.config.kB * np.log(2))  # Convert J/K → bits
    return self.config.nu * S_bits**2 * (1 + sigma * S_bits**self.config.beta)
    
    def fuse_mu(self, mu1, mu2, gamma12):
    """Non-associative fusion: μ1⊗μ2 = μ1 + μ2 + γ12 μ1 μ2"""
    return mu1 + mu2 + gamma12 * mu1 * mu2

    def compute_gamma(self, sigma1, sigma2):
    """Context-dependent coupling coefficient"""
    return self.config.gamma0 * (sigma1 * sigma2) / (sigma1 + sigma2 + 1e-12)

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