import numpy as np
from config import dt, L
from operators import EntropicNS2DOperators  # ou ton nom de classe opérateur

try:
    import cupy as cp
except ImportError:
    cp = None

if __name__ == "__main__":
    print(f"Type check - dt: {type(dt)}, L: {type(L)}")
    print(f"Array shapes - x: {x.shape}, sigma: {sigma.shape}")
    
print(">> Running updated sigma-mu simulation [OK]")


class EntropicNS2DSimulation:
    def __init__(self, config):
        self.config = config
        self.grid = self._init_grid()  # ajoute ça avant
        self.operators = EntropicNS2DOperators(config, self.grid)
        self.fields = self._init_fields()  # Fixed: added underscore
        self.diagnostics = {
            'time': [], 'energy': [], 'enstrophy': [],
            'mean_sigma': [], 'mean_mu': [], 'mean_nstar': []
        }
    
    def _init_grid(self):
        """Initialize physical and spectral grids"""
        x = np.linspace(0, self.config.L, self.config.N, endpoint=False)
        y = np.linspace(0, self.config.L, self.config.N, endpoint=False)
        X, Y = np.meshgrid(x, y)
        
        dx = self.config.L / self.config.N
        kx = 2*np.pi * np.fft.fftfreq(self.config.N, d=dx).reshape(1, -1)
        ky = 2*np.pi * np.fft.fftfreq(self.config.N, d=dx).reshape(-1, 1)
        k2 = kx**2 + ky**2
        k2[0,0] = 1e-12  # Avoid division by zero
        
        if self.config.use_gpu:
            X, Y, kx, ky, k2 = map(cp.asarray, (X, Y, kx, ky, k2))
            
        return {
            'X': X, 'Y': Y, 'x': x, 'y': y,
            'kx': kx, 'ky': ky, 'k2': k2,
            'dx': dx
        }
    
    def _init_fields(self):
        """Initialize physical fields with perturbations"""
        X, Y = self.grid['X'], self.grid['Y']
        noise = 0.01 * np.random.randn(*X.shape)
        
        fields = {
            't': 0.0,  # Initialisé comme float
            'u': np.sin(X) * np.cos(Y) + noise,
            'v': -np.cos(X) * np.sin(Y) + noise,
            'sigma': 0.05 * (np.sin(2*X)**2 + np.cos(3*Y)**2) + 0.01,
            'mu': np.zeros_like(X),
            'n_star': 2.0 * np.ones_like(X),
            't': 0.0,
            'ux': None, 'uy': None,
            'vx': None, 'vy': None
        }
    return fields  
    
    def run(self):
        from time_integration import EntropicNS2DIntegrator
        integrator = EntropicNS2DIntegrator(self.config, self.operators)
        self.fields, self.diagnostics = integrator.run(self.fields, self.diagnostics)

        if self.config.use_gpu:
            for k in fields:
                fields[k] = cp.asarray(fields[k])
                
        return fields