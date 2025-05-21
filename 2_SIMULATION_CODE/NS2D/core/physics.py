import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from scipy.ndimage import gaussian_filter

class PhysicsCore:
    """
    NS2D-TOEND : évolution couplée des champs u, v, sigma, mu, n_star, lambda.
    Tous les couplages dimensionnels et entropiques sont ici centralisés.
    """
    def __init__(self, config, operators):
        self.config = config
        self.ops = operators
        self.xp = cp if (cp is not None and getattr(config, "use_gpu", False)) else np
        self.debug_data = [] if getattr(config, "debug_mode", False) else None

    def update_n_star(self, u, v, sigma):
        """
        Calcule la dimension locale/fractale (n_star) selon la vorticité et l’entropie locale.
        """
        omega = self.ops.curl(u, v)
        omega_x, omega_y = self.ops.grad(omega)
        sigma_x, _ = self.ops.grad(sigma)

        omega_term = self.xp.sqrt(omega_x**2 + omega_y**2)
        sigma_term = self.xp.abs(sigma_x)

        raw_n = 2.3 + 0.5 * (omega_term + 0.3 * sigma_term - 2.3)
        if self.xp == cp:
            raw_n_cpu = cp.asnumpy(raw_n)
            smoothed = gaussian_filter(raw_n_cpu, sigma=1.0)
            smoothed_n = cp.asarray(smoothed)
        else:
            smoothed_n = gaussian_filter(raw_n, sigma=1.0)

        return self.xp.clip(smoothed_n, self.config.nstar_bounds[0], self.config.nstar_bounds[1])

    def entropy_production(self, S, sigma):
        """
        Calcule la production d’entropie (σ) via le taux de strain.
        """
        S_clipped = self.xp.clip(S, 1e-6, self.config.strain_max)
        sigma_clipped = self.xp.clip(sigma, 1e-8, self.config.sigma_max)
        return self.config.nu * S_clipped**2 * (1 + sigma_clipped * S_clipped**self.config.beta)

    def memory_feedback(self, mu, S):
        """
        Calcule le feedback mémoire μ selon strain S.
        """
        S_clip = self.xp.clip(S, 0, self.config.strain_max)
        mu_clip = self.xp.clip(mu, 0, self.config.mu_max)
        return self.config.gamma * mu_clip * self.xp.tanh(S_clip**2 / 0.3)

    def step(self, fields, dt):
        """
        Met à jour tous les champs, intègre la dimension locale, λ, énergie.
        """
        u, v = fields['u'], fields['v']
        sigma, mu = fields['sigma'], fields['mu']

        n_star = self.update_n_star(u, v, sigma)
        S = self.ops.strain_rate(u, v, n_star)
        entropy_prod = self.entropy_production(S, sigma)
        memory_fb = self.memory_feedback(mu, S)

        sigma_new = sigma + dt * (-entropy_prod)
        mu_new = mu + dt * (memory_fb)

        # Placeholders pour la dynamique fluide complète (à compléter si besoin)
        u_new = u
        v_new = v

        # Lambda field (λ) — dynamique, dimensionnelle, TOEND v3
        grad_mu_x, grad_mu_y = self.ops.grad(mu, n_star)
        grad_sigma_x, grad_sigma_y = self.ops.grad(sigma, n_star)
        grad_mu = grad_mu_x + grad_mu_y
        grad_sigma = grad_sigma_x + grad_sigma_y

        sigma_safe = self.xp.clip(sigma, 1e-8, 1e2)
        energy = self.xp.mean(u**2 + v**2)
        energy_safe = energy if energy > 1e-12 else 1.0

        lambda_field = (mu / sigma_safe) / energy_safe \
            + self.config.alpha * grad_mu * grad_sigma \
            + self.config.beta * mu * sigma_safe / energy_safe \
            + 0.05 * n_star  # régulation dimensionnelle

        lambda_field = self.xp.nan_to_num(lambda_field)

        fields_updated = {
            'u': u_new,
            'v': v_new,
            'sigma': sigma_new,
            'mu': mu_new,
            'n_star': n_star,
            'lambda': lambda_field,
            't': fields['t'] + dt
        }

        if self.debug_data is not None:
            self.debug_data.append({
                't': fields['t'],
                'entropy_prod': entropy_prod,
                'memory_fb': memory_fb,
                'n_star': n_star,
                'lambda': lambda_field,
            })

        return fields_updated
