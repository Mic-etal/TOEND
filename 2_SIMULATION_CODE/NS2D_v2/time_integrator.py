import numpy as np
from tqdm import tqdm
from io_utils import save_snapshot
from visualization import plot_fields
from logging_utils import Logger

class EntropicNS2DIntegrator:
    def __init__(self, config, physics, operators):
        self.config = config
        self.physics = physics
        self.ops = operators
        self.dt = config.dt_max
        self.iteration_count = 0
        self.logger = Logger(enabled=not config.debug_mode)

    def compute_adaptive_dt(self, u, v, n_star):
        umax = max(np.abs(u).max(), np.abs(v).max(), 1e-6)
        cfl_dt = self.config.dx / umax
        diff_dt = self.config.dx**2 / (np.abs(n_star).max() + 1e-6)
        return np.clip(min(cfl_dt, diff_dt), self.config.dt_min, self.config.dt_max)

    def run(self, fields):
        t = 0.0
        steps = int(self.config.tmax / self.dt)
        for step in tqdm(range(steps)):
            self.step(fields)
            t += self.dt
            fields['t'] = t
            self.iteration_count += 1

            if step % self.config.save_interval == 0:
                save_snapshot(fields, t, self.config.save_path)
                self.logger.log_scalar("σ_max", np.max(fields["sigma"]), t)
                self.logger.log_scalar("μ_max", np.max(fields["mu"]), t)
                if self.config.debug_mode:
                    plot_fields(fields)

    def step(self, fields):
        u, v = fields["u"], fields["v"]
        sigma, mu = fields["sigma"], fields["mu"]

        # Entropic feedback
        n_star = self.physics.update_n_star(u, v, sigma)
        fields["n_star"] = n_star

        # Placeholder PDE steps (to be replaced with actual dynamics)
        lap_sigma = self.ops.laplace(sigma, n_star)
        S = self.ops.strain_rate(u, v, n_star)
        dsigma = lap_sigma + self.physics.entropy_production(S, sigma)
        dmu = self.physics.memory_feedback(mu, S)

        sigma += self.dt * dsigma
        mu += self.dt * dmu
        fields["sigma"] = sigma
        fields["mu"] = mu

        # Update velocity field
        fields["u"], fields["v"] = self.update_velocity(u, v, sigma, mu)

        # Spectral filtering (if enabled)
        if self.config.use_spectral_filter and self.iteration_count % self.config.filter_interval == 0:
            for key in ["u", "v", "sigma"]:
                fields[key] = self.spectral_filter(fields[key])

        # Adaptive timestep update
        self.dt = self.compute_adaptive_dt(fields["u"], fields["v"], fields["n_star"])

    def update_velocity(self, u, v, sigma, mu):
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
