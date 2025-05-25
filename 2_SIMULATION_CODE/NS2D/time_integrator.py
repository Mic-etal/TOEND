import numpy as np
from tqdm import tqdm
from utils.io_utils import save_snapshot
from utils.visualization import Visualizer
from utils.logging_utils import Logger

class EntropicNS2DIntegrator:
    def __init__(self, config, grid, physics, operators):
        self.config = config
        self.physics = physics
        self.ops = operators
        self.dt = config.dt_max
        self.iteration_count = 0
        self.logger = Logger.setup_logger(enabled=not config.debug_mode)
        self.grid = grid

    def compute_adaptive_dt(self, u, v, n_star):
        umax = max(np.abs(u).max(), np.abs(v).max(), 1e-6)
        cfl_dt = self.config.dx / umax
        diff_dt = self.config.dx**2 / (np.abs(n_star).max() + 1e-6)
        return np.clip(min(cfl_dt, diff_dt), self.config.dt_min, self.config.dt_max)

    def run(self, fields):
        t = 0.0
        steps = int(self.config.tmax / self.dt)
        diagnostics = {
            'uncertainty_sigma': [],
            'memory_mu': [],
            'entropy_S': [],
            'lambda': [],
            'dissipation_dirr': [],
            # ajoute ici d'autres diagnostics si besoin (n_star, energy, ...)
        }
        for step in tqdm(range(steps)):
            self.step(fields)
            t += self.dt
            fields['t'] = t
            self.iteration_count += 1

            # Diagnostic TOEND
            self.record_diagnostics(fields, diagnostics, self.dt)

            if step % self.config.save_interval == 0:
                save_snapshot(fields, self.grid, self.config)
                self.logger.info(f"[t={t:.3f}] sigma_max={np.max(fields['sigma']):.4f}")
                self.logger.info(f"[t={t:.3f}] mu_max={np.max(fields['mu']):.4f}")
                if self.config.debug_mode:
                    plot_fields(fields)

        # Exporte les diagnostics à la fin si besoin (ex: np.savez, CSV, etc.)
        np.savez_compressed(f"{self.config.save_path}/diagnostics_TOEND.npz", **diagnostics)

    def step(self, fields):
        u, v = fields["u"], fields["v"]
        sigma, mu = fields["sigma"], fields["mu"]

        # Entropic feedback
        n_star = self.physics.update_n_star(u, v, sigma)
        fields["n_star"] = n_star

        # Placeholder PDE steps (à compléter avec ta vraie dynamique NS2D !)
        lap_sigma = self.ops.laplace(sigma, n_star)
        S = self.ops.strain_rate(u, v, n_star)
        dsigma = lap_sigma + self.physics.entropy_production(S, sigma)
        dmu = self.physics.memory_feedback(mu, S)

        sigma += self.dt * dsigma
        mu += self.dt * dmu
        fields["sigma"] = sigma
        fields["mu"] = mu

        # Update velocity field (placeholder)
        fields["u"], fields["v"] = self.update_velocity(u, v, sigma, mu)

        # Spectral filtering (if enabled)
        if self.config.use_spectral_filter and self.iteration_count % self.config.filter_interval == 0:
            for key in ["u", "v", "sigma"]:
                fields[key] = self.spectral_filter(fields[key])

        # Adaptive timestep update
        self.dt = self.compute_adaptive_dt(fields["u"], fields["v"], fields["n_star"])

    def update_velocity(self, u, v, sigma, mu):
        # Placeholder: à remplacer par la dynamique fluide NS2D complète
        ux, uy = self.ops.grad(u)
        vx, vy = self.ops.grad(v)
        curl_term = self.ops.curl(u, v)
        damping = -0.01 * curl_term
        return u + self.dt * damping, v + self.dt * damping

    def spectral_filter(self, field):
        field_hat = np.fft.fft2(field)
        kx = self.ops.grid["kx"]
        ky = self.ops.grid["ky"]
        k2 = kx**2 + ky**2
        mask = k2 < self.config.k_cutoff**2
        return np.fft.ifft2(field_hat * mask).real

    def record_diagnostics(self, fields, diagnostics, dt):
        """
        Enregistre tous les diagnostics scalaires TOEND à chaque pas.
        - σ, μ, S, λ, Dirr (dissipation info)
        """
        sigma_val = np.mean(fields['sigma'])
        mu_val = np.mean(fields['mu'])
        S_val = sigma_val + mu_val

        diagnostics.setdefault('uncertainty_sigma', []).append(float(sigma_val))
        diagnostics.setdefault('memory_mu', []).append(float(mu_val))
        diagnostics.setdefault('entropy_S', []).append(float(S_val))
        
        # Lambda (λ)
        if 'lambda' in fields:
            lambda_val = np.mean(fields['lambda'])
        else:
            # fallback : approx λ = dμ/dσ
            if len(diagnostics['memory_mu']) >= 2 and len(diagnostics['uncertainty_sigma']) >= 2:
                dmu = diagnostics['memory_mu'][-1] - diagnostics['memory_mu'][-2]
                dsigma = diagnostics['uncertainty_sigma'][-1] - diagnostics['uncertainty_sigma'][-2]
                lambda_val = dmu / dsigma if dsigma != 0 else 0.0
            else:
                lambda_val = 0.0
        diagnostics.setdefault('lambda', []).append(lambda_val)
        
        # Dissipation (Dirr)
        if len(diagnostics['entropy_S']) >= 2:
            dS = diagnostics['entropy_S'][-1] - diagnostics['entropy_S'][-2]
            dissipation = -dS / dt
        else:
            dissipation = 0.0
        diagnostics.setdefault('dissipation_dirr', []).append(dissipation)

