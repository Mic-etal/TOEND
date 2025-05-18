import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from scipy.ndimage import gaussian_filter
import os
# config.py
from dataclasses import dataclass

# Domain and resolution
N = 256
L = 10.0
dx = L / N
# Décroissance de σ (à vérifier selon ton modèle)
lam = 0.1  # ou lambda_sigma si tu préfères ce nom

# Time
t_max = 5.0
dt = 0.01  # ← AJOUTE CETTE LIGNE SI ABSENTE

@dataclass
class SimulationConfig:
    tmax: float = 10.0    # Doit être un float
    # Grid parameters
    dt = 0.01                  # Must be float
    N: int = 256          # Grid size
    L: float = 2*np.pi    # Domain size
    
    # Physics parameters
    alpha: float = 0.3    # Sigma injection coefficient
    eta: float = 0.1     # Sigma diffusion coefficient
    lmbda: float = 0.1    # Sigma decay coefficient
    nu: float = 0.001     # Viscosity
    beta: float = 0.2     # Non-linearity exponent
    gamma: float = 0.5   # Memory feedback strength
    # Nouveaux paramètres vortex
    kappa = 0.02  # Couplage vortex
    mu_max = 1.0     # Maximum μ capacity
    
    # Time parameters
    tmax: float = 10.0    # Total simulation time
    dt_max: float = 0.1  # Maximum timestep
    dt_min: float = 1e-6  # Minimum timestep (nouveau)
    dx = L / N          # Grid spacing
    # Paramètres de bruit
    add_noise = True
    noise_strength = 0.01  # Intensité du bruit
    
    # n* parameters
    nstar_bounds: tuple = (1.8, 2.8)  # Min/max n* values
    
    # Stabilization options (nouveaux)
    use_spectral_filter: bool = True   # Activer le filtrage spectral
    filter_interval: int = 10         # Intervalle de filtrage
    strict_clipping: bool = True      # Activer le clipping strict
    k_cutoff = 0.8 * (N//2)     # Cutoff wavenumber
    
    # Runtime options
    use_gpu: bool = False
    save_interval = 100  # Pas de sauvegarde
    save_path: str = "results"
    
    # Nouveaux paramètres de stabilisation
    strain_max: float = 10.0          # Valeur maximale pour le strain
    mu_max: float = 5.0               # Valeur maximale pour mu
    sigma_max: float = 20.0           # Valeur maximale pour sigma
    dt_min: float = 1e-5              # Pas de temps minimum absolu
    debug_mode: bool = False          # Mode diagnostic détaillé
    # SIMULATION MODE ###################################
mode = "fluid"  # Options: ["fluid", "sigma_mu"]

# σ-μ PARAMETERS ####################################
# Paramètres physiques
eta = 0.005       # Diffusion
alpha = 0.5       # Injection
lam = 0.1         # Décroissance
beta = 0.5        # Non-linéarité
gamma = 0.3       # Croissance mémoire
mu_max = 1.0      # Capacité mémoire
kappa = 0.1       # Sensibilité au gradient

# Paramètres numériques
N = 256           # Points de grille
L = 10.0          # Taille du domaine
dt = 0.01         # Pas de temps
t_max = 5.0       # Durée simulation
boundary_epsilon = 1e-6  # Valeur aux bords
sigma_max = 2.0   # Limite supérieure σ