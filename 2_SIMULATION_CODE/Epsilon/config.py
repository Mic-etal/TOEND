import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
import os

try:
    import cupy as cp
except ImportError:
    cp = None

@dataclass
class SimulationConfig:
    # Time parameters
    tmax: float = 5.0
    dt: float = 0.01
    
    # Domain and resolution
    N: int = 256
    L: float = 10.0  # Changed from 2π to match your domain setup
    dx: float = L / N  # Added missing parameter
    
    # Physical parameters
    alpha: float = 0.5
    eta: float = 0.005
    lmbda: float = 0.1
    nu: float = 0.001
    beta: float = 0.5
    gamma: float = 0.3
    mu_max: float = 1.0
    kappa: float = 0.1  # Added missing parameter
    
    # Noise parameters
    add_noise: bool = True
    noise_strength: float = 0.01
    
    # n* parameters
    nstar_bounds: tuple = (1.8, 2.8)
    
    # Stabilization
    use_spectral_filter: bool = True
    filter_interval: int = 10
    strict_clipping: bool = True
    k_cutoff: float = 0.8 * (N//2)
    
    # Runtime options
    use_gpu: bool = False
    save_interval: int = 100
    save_path: str = "results"
    
    # Stability thresholds
    strain_max: float = 10.0
    sigma_max: float = 2.0
    dt_min: float = 1e-5
    debug_mode: bool = False

# Instantiate config with GPU override
config = SimulationConfig(
    use_gpu=True,  # Override default GPU setting
    L=10.0,        # Explicitly set domain size
    dx=10.0/256    # Ensure consistency
)

# Simulation mode (external variable)
mode = "fluid"  # Options: ["fluid", "sigma_mu"]