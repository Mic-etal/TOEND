import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from scipy.ndimage import gaussian_filter

class PhysicsCore:
    """
    Noyau physique pour la simulation NS2D entropique TOEND.

    Intègre les mises à jour dynamiques de n_star, la production d'entropie et le feedback mémoire.
    """

    def __init__(self, config, operators):
        self.config = config
        self.ops = operators
        self.debug_data = [] if getattr(config, "debug_mode", False) else None
        self.xp = cp if (cp is not None and getattr(config, "use_gpu", False)) else np

    def update_n_star(self, u, v, sigma):
        """
        Met à jour la dimension fractale locale n_star en fonction de la vorticité et de sigma.

        Args:
            u, v : champs vitesse
            sigma : champ d'entropie locale

        Returns:
            n_star : champ dimension fractale locale mis à jour, borné
        """
        omega = self.ops.curl(u, v)
        omega_x, omega_y = self.ops.grad(omega)
        sigma_x, _ = self.ops.grad(sigma)

        omega_term = self.xp.sqrt(omega_x**2 + omega_y**2)
        sigma_term = self.xp.abs(sigma_x)

        raw_n = 2.3 + 0.5 * (omega_term + 0.3 * sigma_term - 2.3)
        # Passage CPU/GPU pour gaussian_filter (scipy pas GPU)
        if self.xp == cp:
            raw_n_cpu = cp.asnumpy(raw_n)
            smoothed = gaussian_filter(raw_n_cpu, sigma=1.0)
            smoothed_n = cp.asarray(smoothed)
        else:
            smoothed_n = gaussian_filter(raw_n, sigma=1.0)

        return self.xp.clip(smoothed_n, self.config.nstar_bounds[0], self.config.nstar_bounds[1])

    def entropy_production(self, S, sigma):
        """
        Calcule la production d'entropie à partir du taux de déformation S et sigma.

        Args:
            S : taux de déformation (strain rate)
            sigma : champ entropie locale

        Returns:
            Terme de production d'entropie
        """
        S_clipped = self.xp.clip(S, 1e-6, self.config.strain_max)
        sigma_clipped = self.xp.clip(sigma, 1e-8, self.config.sigma_max)
        return self.config.nu * S_clipped**2 * (1 + sigma_clipped * S_clipped**self.config.beta)

    def memory_feedback(self, mu, S):
        """
        Calcule le feedback de mémoire basé sur mu et le taux de déformation S.

        Args:
            mu : champ mémoire locale
            S : taux de déformation (strain rate)

        Returns:
            Terme de feedback mémoire
        """
        S_clip = self.xp.clip(S, 0, self.config.strain_max)
        mu_clip = self.xp.clip(mu, 0, self.config.mu_max)
        return self.config.gamma * mu_clip * self.xp.tanh(S_clip**2 / 0.3)

    def step(self, fields, dt):
        """
        Une étape de simulation complète : mise à jour des champs u, v, sigma, mu, n_star.

        Args:
            fields (dict): champs actuels de la simulation
            dt (float): pas de temps

        Returns:
            dict: champs mis à jour
        """
        u, v = fields['u'], fields['v']
        sigma, mu = fields['sigma'], fields['mu']

        n_star = self.update_n_star(u, v, sigma)

        S = self.ops.strain_rate(u, v, n_star)
        entropy_prod = self.entropy_production(S, sigma)
        memory_fb = self.memory_feedback(mu, S)

        sigma_new = sigma + dt * (-entropy_prod)
        mu_new = mu + dt * (memory_fb)

        # Vitesse et autres champs peuvent être mis à jour ici (simplifié)
        u_new = u  # Placeholder : à implémenter la dynamique NS2D complète
        v_new = v

        fields_updated = {
            'u': u_new,
            'v': v_new,
            'sigma': sigma_new,
            'mu': mu_new,
            'n_star': n_star,
            't': fields['t'] + dt
        }

        if self.debug_data is not None:
            self.debug_data.append({
                't': fields['t'],
                'entropy_prod': entropy_prod,
                'memory_fb': memory_fb,
                'n_star': n_star
            })

        return fields_updated
