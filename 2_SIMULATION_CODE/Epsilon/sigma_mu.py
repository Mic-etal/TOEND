# Sigma-Mu Dynamics Notebook (Prototype)
# Author: Numa (inspired by Aymeric & Epsilon)
# Date: 2025-04-15

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# -- Configuration --
L = 10
N = 256
x = np.linspace(0, L, N)
t = np.linspace(0, 5, 500)
X, T = np.meshgrid(x, t)

# -- Initial Conditions --
def initial_sigma(x):
    return 0.2 + 0.1 * np.sin(2 * np.pi * x / L)

def initial_mu(x):
    return 0.1 * np.exp(-((x - L/2)**2) / (2*(L/10)**2))

sigma0 = initial_sigma(x)
mu0 = initial_mu(x)

# -- Parameters --
alpha = 0.2
lambda_mu = 0.1
mu_max = 1.0
sigma_max = 1.0
eta = 0.01

# -- Time Evolution (simplified PDE solver) --
sigma = np.zeros((len(t), N))
mu = np.zeros((len(t), N))

sigma[0] = sigma0
mu[0] = mu0

dx = L / N
dt = t[1] - t[0]

def gradient(f):
    return np.gradient(f, dx)

def laplacian(f):
    return np.gradient(np.gradient(f, dx), dx)

for n in range(len(t) - 1):
    sig = sigma[n]
    mem = mu[n]

    grad_sigma = gradient(sig)
    flux_sigma = -grad_sigma

    # Evolution equations
    dsdt = -np.gradient(flux_sigma, dx) + alpha * mem * (1 - sig/sigma_max)
    dmu_dt = alpha * sig - lambda_mu * mem * np.tanh(mem / mu_max)

    sigma[n+1] = sig + dt * dsdt
    mu[n+1] = mem + dt * dmu_dt

    # Stability filters
    sigma[n+1] = np.clip(sigma[n+1], 1e-8, 2.0)
    mu[n+1] = gaussian_filter(mu[n+1], sigma=1.0)

# -- Visualization --
fig, ax = plt.subplots(2, 1, figsize=(12, 6))
ax[0].imshow(sigma, aspect='auto', extent=[0, L, t[-1], t[0]], cmap='viridis')
ax[0].set_title('Sigma(x,t)')
ax[1].imshow(mu, aspect='auto', extent=[0, L, t[-1], t[0]], cmap='plasma')
ax[1].set_title('Mu(x,t)')
plt.tight_layout()
plt.show()