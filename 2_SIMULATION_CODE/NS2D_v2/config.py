import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from dataclasses import dataclass

# -- DOMAIN PARAMETERS --
N = 256
L = 10.0
DX = L / N

# -- SIMULATION MODES --
mode = "fluid"  # Options: ["fluid", "sigma_mu"]

# -- CONFIG FOR NS2D FLUID DYNAMICS --
@dataclass
class ConfigNS2D:
    N: int = N
    L: float = L
    dx: float = DX
    dt: float = 0.01
    tmax: float = 10.0
    
    # Entropic dynamics
    alpha: float = 0.3
    eta: float = 0.1
    lmbda: float = 0.1
    nu: float = 0.001
    beta: float = 0.2
    gamma: float = 0.5
    kappa: float = 0.02
    mu_max: float = 5.0
    
    # Stability and filtering
    use_spectral_filter: bool = True
    filter_interval: int = 10
    strict_clipping: bool = True
    k_cutoff: float = 0.8 * (N // 2)
    
    # Noise and runtime
    add_noise: bool = True
    noise_strength: float = 0.01
    save_interval: int = 100
    save_path: str = "results"
    use_gpu: bool = False
    debug_mode: bool = False
    
    # CFL and strain control
    dt_max: float = 0.1
    dt_min: float = 1e-5
    strain_max: float = 10.0
    sigma_max: float = 20.0

    # n* modulation
    nstar_bounds: tuple = (1.8, 2.8)

# -- CONFIG FOR SIGMA-MU DYNAMICS (1D) --
@dataclass
class ConfigSigmaMu:
    N: int = 256
    L: float = 10.0
    dx: float = L / 256
    dt: float = 0.01
    tmax: float = 5.0
    
    alpha: float = 0.5
    eta: float = 0.005
    lmbda: float = 0.1
    beta: float = 0.5
    gamma: float = 0.3
    kappa: float = 0.1
    mu_max: float = 1.0
    sigma_max: float = 2.0
    boundary_epsilon: float = 1e-6
    
    add_noise: bool = False
    save_interval: int = 50
    save_path: str = "results_1d"
    debug_mode: bool = False
