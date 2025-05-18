import numpy as np
class SimulationValidator:
    def __init__(self, config):
        self.config = config
        self.crash_count = 0
        
def compute_divergence(self, u, v):
    """Compute divergence field with proper normalization"""
    dx = self.config.L / self.config.N
    
    if self.config.use_gpu:
        u_cpu, v_cpu = cp.asnumpy(u), cp.asnumpy(v)
    else:
        u_cpu, v_cpu = u, v
    
    # Calcul plus précis de la divergence
    ux = np.gradient(u_cpu, dx, axis=1)
    vy = np.gradient(v_cpu, dx, axis=0)
    div = ux + vy
    
    # Normalisation par l'échelle de vitesse caractéristique
    char_velocity = np.sqrt(np.mean(u_cpu**2 + v_cpu**2))
    normalized_div = div / (char_velocity + 1e-8) * self.config.L
    
    return normalized_div

        
    def check_conservation(self, diagnostics, tol=1e-3):
        """Verify energy conservation within tolerance"""
        energy = np.array(diagnostics['energy'])
        max_diff = np.abs(energy.max() - energy.min())
        return max_diff < tol * energy.mean()

    def check_cfl_violation(self, diagnostics):
        """Check if CFL condition was ever violated"""
        dt = np.diff(diagnostics['time'])
        dx = self.config.L / self.config.N
        u_max = np.sqrt(2 * np.array(diagnostics['energy']))
        cfl = dt * u_max / dx
        return np.any(cfl > 0.5)  # Standard CFL threshold

        # validation.py - Nouveau module
def check_energy_balance(diagnostics):
    """Vérifie la conservation d'énergie avec tolérance numérique."""
    energy = np.array(diagnostics['energy'])
    rel_diff = np.abs(energy.max() - energy.min()) / energy.mean()
    return rel_diff < 0.1  # Tolérance 10%

def verify_gradients(fields, grid):
    """Teste la cohérence des opérateurs différentiels."""
    u, v = fields['u'], fields['v']
    div_curl = div(grad(curl(u, v)))  # Doit être ~0 théoriquement
    return np.mean(np.abs(div_curl)) < 1e-6
    
    
    def validate_fields(self, fields):
        """Comprehensive field validation"""
        checks = [
            self.check_nan_inf(fields),
            self.check_divergence(fields['u'], fields['v']),
            self.check_bounds(fields)
        ]
        return all(checks)

    def check_bounds(self, fields):
        """Check all physical bounds"""
        valid = True
        
        # Vérification sigma
        sigma = fields['sigma']
        if np.any(sigma < 0) or np.any(sigma > self.config.sigma_max):
            print(f"Sigma out of bounds [0, {self.config.sigma_max}]")
            valid = False
            
        # Vérification mu
        mu = fields['mu']
        if np.any(mu < 0) or np.any(mu > self.config.mu_max):
            print(f"Mu out of bounds [0, {self.config.mu_max}]")
            valid = False
            
        return valid
        
def generate_report(data):
    """Génère un rapport PDF avec:
    - Évolution temporelle des grandeurs
    - Diagramme de phase final
    - Paramètres utilisés
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages('validation_report.pdf') as pdf:
        # Plot 1: Énergie de σ
        plt.figure()
        plt.plot(data['sigma_history'])
        plt.title("Évolution de σ")
        pdf.savefig()
        plt.close()
        
        # Plot 2: Portrait de phase
        plt.figure()
        plt.scatter(data['sigma'][-1], data['mu'][-1], alpha=0.5)
        plt.xlabel("σ"); plt.ylabel("μ")
        pdf.savefig()
        plt.close()