import numpy as np
from config.config import ConfigNS2D, ConfigSigmaMu
from core.operators import EntropicOperators
from core.physics import PhysicsCore
from time_integrator import EntropicNS2DIntegrator


class EntropicNS2DSimulation:
    def __init__(self, config: ConfigNS2D):
        self.config = config
        self.grid, self.kx, self.ky = self._init_grid()
        self.fields = self._init_fields()
        self.ops = EntropicOperators(config, self.grid, mode="2D")
        self.physics = PhysicsCore(config, self.ops)
        self.integrator = EntropicNS2DIntegrator(config, self.grid, self.physics, self.ops)

    def _init_grid(self):
        k = np.fft.fftfreq(self.config.N, d=self.config.dx) * 2 * np.pi
        kx, ky = np.meshgrid(k, k, indexing="ij")
        k2 = kx**2 + ky**2 + 1e-8
        return {"kx": kx, "ky": ky, "k2": k2}, kx, ky

    def _init_fields(self):
        shape = (self.config.N, self.config.N)
        u = np.zeros(shape)
        v = np.zeros(shape)
        sigma = np.random.rand(*shape) * 0.1
        mu = np.zeros_like(sigma)
        n_star = np.ones_like(sigma) * 2.0
        return {"u": u, "v": v, "sigma": sigma, "mu": mu, "n_star": n_star, "t": 0.0}

    def run(self):
        print("[NS2D] Simulation started")
        return self.integrator.run(self.fields)   


class SigmaMu1DSimulation:
    def __init__(self, config: ConfigSigmaMu):
        self.config = config
        self.x = np.linspace(0, config.L, config.N)
        self.sigma = np.ones_like(self.x)
        self.mu = np.zeros_like(self.x)
        self.t = 0.0

    def update_sigma(self):
        lap_sigma = np.gradient(np.gradient(self.sigma, self.config.dx), self.config.dx)
        injection = self.config.alpha * (1 - self.sigma) - self.config.eta * self.mu * self.sigma
        return lap_sigma + injection

    def update_mu(self):
        return self.config.gamma * self.mu * (self.sigma - 1)

    def run(self):
        print("[Sigma-Mu 1D] Simulation started")
        steps = int(self.config.tmax / self.config.dt)
        for step in range(steps):
            dsigma = self.update_sigma()
            dmu = self.update_mu()
            self.sigma += self.config.dt * dsigma
            self.mu += self.config.dt * dmu
            self.t += self.config.dt
            if step % self.config.save_interval == 0:
                print(f"[t={self.t:.2f}] max(σ)={np.max(self.sigma):.2f}, max(μ)={np.max(self.mu):.2f}")
        # On retourne directement les champs
        return {'x': self.x, 'sigma': self.sigma, 'mu': self.mu, 't': self.t}, {}
        

class EntropicSimulator:
    def __init__(self, config):
        if config.mode == "fluid":
            self.sim = EntropicNS2DSimulation(config)
        elif config.mode == "sigma_mu":
            self.sim = SigmaMu1DSimulation(config)
        else:
            raise ValueError("Unknown simulation mode")

    def run(self):
        return self.sim.run()