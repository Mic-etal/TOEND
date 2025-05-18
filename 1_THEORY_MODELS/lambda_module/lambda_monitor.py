"""
Lambda Monitor – TOEND Module

This module computes the λ value given inputs µ (coherence), σ (entropy), and E (energy).
Optional parameters alpha and beta allow modulation of contradiction and saturation terms.

Usage:
    lambda_value = compute_lambda(mu=0.5, sigma=0.2, energy=1.0)
"""

def compute_lambda(mu, sigma, energy, alpha=1.0, beta=1.0, grad_mu=0.0, grad_sigma=0.0):
    """Compute the lambda value from TOEND model inputs."""
    if sigma == 0 or energy == 0:
        raise ValueError("σ and E must be non-zero to avoid division errors.")

    sensitivity_term = (mu / sigma) * (1 / energy)
    contradiction_term = alpha * grad_mu * grad_sigma
    saturation_term = beta * mu * sigma / energy

    lambda_val = sensitivity_term + contradiction_term + saturation_term
    return lambda_val

# Example usage
if __name__ == "__main__":
    mu = 0.5
    sigma = 0.2
    energy = 1.0
    # Explicitly specify alpha and beta to avoid confusion with grad_mu/grad_sigma
    lambda_result = compute_lambda(mu, sigma, energy, alpha=1.0, beta=1.0, grad_mu=0.1, grad_sigma=0.2)
    print("Computed λ =", lambda_result)