import numpy as np
from config import *
from visualization import plot_phase_portrait, plot_field_evolution
import time

class SigmaMuSystem:
    def __init__(self):
        # Initialisation du domaine 1D
        self.dx = L / N
        self.x = np.linspace(0, L, N)
        
        # Conditions initiales
        self.sigma = 0.1 * np.exp(-(self.x-L/2)**2/(0.2*L)**2) + 0.01*np.random.rand(N)
        self.mu = np.zeros_like(self.sigma)
        self.time = 0.0
        
        # Historique
        self.history = {
            'time': [],
            'sigma': [],
            'mu': [],
            'grad_sigma': [],
            'correlation': []
        }

    def compute_gradients(self):
        """Calcule les gradients avec régularisation"""
        grad_sigma = np.gradient(self.sigma, self.dx)
        return np.abs(grad_sigma) + 1e-12  # S régularisé

    def update_sigma(self, S):
        """Évolution de l'entropie σ"""
        diffusion = eta * np.gradient(np.gradient(self.sigma, self.dx), self.dx)
        injection = alpha * np.sqrt(np.maximum(self.sigma, 1e-8)) * S**1.5
        decay = -lam * self.sigma * S**(2 - beta)
        return self.sigma + dt * (diffusion + injection + decay)

    def update_mu(self, S):
        """Évolution de la mémoire μ avec γ adaptatif"""
        S_thresh = np.median(S)  # Seuil dynamique
        gamma_eff = gamma * (1 + np.tanh(S - S_thresh))  # γ adaptatif
        return self.mu + dt * gamma_eff * self.sigma * (1 - self.mu/mu_max) * (1 + kappa * S)

    def apply_boundary_conditions(self):
        """Conditions aux limites absorbantes"""
        self.sigma[0] = self.sigma[-1] = boundary_epsilon
        self.mu[0] = self.mu[-1] = 0.0

    def record_history(self):
        """Enregistre l'état courant"""
        if len(self.history['time']) % 10 == 0:  # Tous les 10 pas
            S = self.compute_gradients()
            self.history['time'].append(self.time)
            self.history['sigma'].append(self.sigma.copy())
            self.history['mu'].append(self.mu.copy())
            self.history['grad_sigma'].append(S.copy())
            
            # Calcul de la corrélation σ-μ
            valid_mask = (self.sigma > 1e-3) & (self.mu > 1e-3)
            if np.any(valid_mask):
                corr = np.corrcoef(self.sigma[valid_mask], self.mu[valid_mask])[0,1]
                self.history['correlation'].append(corr)

    def run(self):
        """Boucle principale de simulation"""
        try:
            while self.time < t_max:
                S = self.compute_gradients()
                
                # Mise à jour des champs
                self.sigma = self.update_sigma(S)
                self.mu = self.update_mu(S)
                self.apply_boundary_conditions()
                
                # Contraintes physiques
                self.sigma = np.clip(self.sigma, 1e-6, sigma_max)
                self.mu = np.clip(self.mu, 0.0, mu_max)
                
                # Enregistrement et visualisation
                self.record_history()
                if len(self.history['time']) % 100 == 0:
                    plot_field_evolution(self.x, self.sigma, self.mu, self.time)
                
                self.time += dt
                
        except Exception as e:
            print(f"Crash à t={self.time:.2f}: {str(e)}")
            self.save_state()
            raise
        
        self.save_results()

    def save_state(self):
        """Sauvegarde l'état actuel"""
        np.savez(f'crash_state_{time.time()}.npz',
                 x=self.x,
                 sigma=self.sigma,
                 mu=self.mu,
                 last_time=self.time)

    def save_results(self):
        """Sauvegarde les résultats finaux"""
        results = {
            'params': {
                'eta': eta, 'alpha': alpha, 'lam': lam,
                'beta': beta, 'gamma': gamma, 'mu_max': mu_max,
                'kappa': kappa, 'boundary_epsilon': boundary_epsilon
            },
            'x': self.x,
            **self.history
        }
        np.savez('final_results.npz', **results)
        plot_phase_portrait(self.history['sigma'], self.history['mu'])


def run_sigma_mu():
    system = SigmaMuSystem()
    system.run()
    return system.history