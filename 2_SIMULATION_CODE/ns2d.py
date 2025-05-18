import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft2, ifft2, fftfreq
from scipy.linalg import eig
from scipy.sparse.linalg import svds
import pywt
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class EntropicNavierStokes2D:
    def __init__(self, N=256, L=2*np.pi, Re=5000, t_max=10.0, save_dir='results'):
        """Initialize the 2D Entropic Navier-Stokes solver with Koopman analysis"""
        # Physics parameters
        self.N, self.L, self.Re, self.t_max = N, L, Re, t_max
        self.nu = 1.0 / Re
        self.eta, self.alpha, self.beta, self.gamma = 0.1, 0.8, 0.4, 0.3
        self.collapse_thresh, self.collapse_strength = 0.9, 0.5

        # Domain setup
        self.x = np.linspace(0, L, N, endpoint=False)
        self.y = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.dx = self.x[1] - self.x[0]
        self.dt = 0.002 * self.dx  # Conservative CFL
        self.t = 0

        # Spectral operators
        self.kx = 2*np.pi*fftfreq(N, d=self.dx).reshape(1, N)
        self.ky = 2*np.pi*fftfreq(N, d=self.dx).reshape(N, 1)
        self.k_sq = self.kx**2 + self.ky**2
        self.k_sq[0, 0] = 1  # Avoid division by zero

        # Initialize fields (Taylor-Green vortex + noise)
        self.u = np.sin(self.X)*np.cos(self.Y) + 0.15*np.random.randn(N,N)
        self.v = -np.cos(self.X)*np.sin(self.Y) + 0.15*np.random.randn(N,N)
        self.sigma = 0.1*(np.sin(2*self.X)**2 + np.cos(3*self.Y)**2) + 0.02
        self.mu = np.zeros((N,N))

        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f"run_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)

        # Enhanced diagnostics
        self.history = {
            't': [], 'energy': [], 'enstrophy': [], 
            'entropy_prod': [], 'max_strain': [], 'mean_sigma': [],
            'entropic_ratio': [], 'koopman_eigs': []
        }
        self.snapshots = []
        self.snapshot_times = []
        self.next_snapshot_time = 0
        self.snapshot_interval = 0.1

    # Core physics methods
    def spectral_grad(self, f):
        """Compute gradient using spectral differentiation"""
        return ifft2(1j*self.kx*fft2(f)).real, ifft2(1j*self.ky*fft2(f)).real

    def compute_vorticity(self):
        """Vorticity ω = ∂xv - ∂yu with spectral accuracy"""
        return ifft2(1j*self.kx*fft2(self.v) - 1j*self.ky*fft2(self.u)).real

    def compute_strain_rate(self):
        """Strain rate tensor norm |S|"""
        du_dx, du_dy = self.spectral_grad(self.u)
        dv_dx, dv_dy = self.spectral_grad(self.v)
        return np.sqrt(2*(du_dx**2 + dv_dy**2 + 0.5*(du_dy + dv_dx)**2))

    def entropy_production_rate(self, strain):
        """Modified entropy production with intermittency"""
        return self.nu * strain**2 * (1 + self.sigma*strain**self.beta/(1 + 0.2*strain))

    def memory_feedback(self, strain):
        """Nonlinear memory coupling with saturation"""
        return self.gamma * self.mu * np.tanh(strain**2 / 0.3)

    def spectral_filter(self, f):
        """8th-order exponential spectral filter for dealiasing"""
        k_cut = 0.66 * np.max(np.abs(self.kx))
        filt = np.exp(-(np.sqrt(self.k_sq)/k_cut)**8)
        return ifft2(fft2(f) * filt).real

    def apply_collapse(self, strain):
        """⊠-transition: localized collapse of uncertainty to memory"""
        collapse_mask = (self.sigma > self.collapse_thresh*self.sigma.max()) & (strain > 0.5*np.max(strain))
        if np.any(collapse_mask):
            memory_gain = self.collapse_strength * self.sigma * collapse_mask
            self.sigma *= (1 - self.collapse_strength * collapse_mask)
            self.mu += memory_gain
            self.save_snapshot("collapse")

    # Time integration
    def evolve(self):
        """Run simulation with adaptive diagnostics"""
        pbar = tqdm(total=int(self.t_max/self.dt))
        while self.t < self.t_max:
            strain = self.compute_strain_rate()
            omega = self.compute_vorticity()
            
            # --- Entropic Memory Update ---
            self.mu += self.entropy_production_rate(strain) * self.dt
            
            # --- Uncertainty Dynamics ---
            lap_sigma = ifft2(-self.k_sq*fft2(self.sigma)).real
            decay = -0.15 * self.sigma * strain**(2.2 - self.beta)
            injection = self.alpha * np.sqrt(np.maximum(self.sigma, 0)) * strain**1.7
            self.sigma += (self.eta*lap_sigma + decay + injection) * self.dt
            self.sigma = np.maximum(self.sigma, 1e-8)
            
            # --- ⊠-Transition Event ---
            if self.t > 0.3*self.t_max:
                self.apply_collapse(strain)
            
            # --- Navier-Stokes with Memory Feedback ---
            # Nonlinear terms (conservative form)
            u_conv = -0.5*(ifft2(1j*self.kx*fft2(self.u**2)).real + 
                          ifft2(1j*self.ky*fft2(self.u*self.v)).real)
            v_conv = -0.5*(ifft2(1j*self.kx*fft2(self.u*self.v)).real + 
                          ifft2(1j*self.ky*fft2(self.v**2)).real)
            
            # Viscous terms
            u_visc = self.nu * ifft2(-self.k_sq*fft2(self.u)).real
            v_visc = self.nu * ifft2(-self.k_sq*fft2(self.v)).real
            
            # Memory feedback
            grad_mu_x, grad_mu_y = self.spectral_grad(self.mu)
            Gamma = self.memory_feedback(strain)
            u_mem = -Gamma * grad_mu_x
            v_mem = -Gamma * grad_mu_y
            
            # Time integration
            self.u += (u_conv + u_visc + u_mem) * self.dt
            self.v += (v_conv + v_visc + v_mem) * self.dt
            
            # Spectral filtering
            self.u = self.spectral_filter(self.u)
            self.v = self.spectral_filter(self.v)
            self.sigma = self.spectral_filter(self.sigma)
            
            # Diagnostics and snapshots
            self._log_diagnostics(strain, omega)
            if self.t >= self.next_snapshot_time:
                self.save_snapshot()
                self.next_snapshot_time += self.snapshot_interval
            
            self.t += self.dt
            pbar.update(1)
        pbar.close()
        self.save_final_state()
        self.analyze_results()

    # Analysis methods
    def koopman_analysis(self, num_modes=5):
        """Dynamic Mode Decomposition of (u, σ, μ) fields"""
        # Prepare data matrices
        data = np.array([np.stack([s['u'], s['sigma'], s['mu']], axis=0) 
                         for s in self.snapshots])  # [time, 3, N, N]
        X = data[:-1].reshape(len(self.snapshots)-1, -1)
        Y = data[1:].reshape(len(self.snapshots)-1, -1)
        
        # SVD-based DMD
        U, s, Vh = svds(X, k=num_modes)
        K = (Y.T @ U) @ np.diag(1/s) @ Vh  # Koopman approximation
        
        # Eigenanalysis
        eigvals, eigmodes = eig(K.T)
        modes = eigmodes[:, :num_modes].reshape(3, self.N, self.N, num_modes)
        
        # Store eigenvalues
        self.history['koopman_eigs'].append(eigvals[:num_modes])
        
        # Plot interactive modes
        fig = make_subplots(rows=num_modes, cols=3,
                          subplot_titles=[f"Mode {i+1} (λ={np.abs(eigvals[i]):.3f})" 
                                         for i in range(num_modes)])
        
        for i in range(num_modes):
            fig.add_trace(go.Heatmap(z=modes[0,...,i], colorscale='RdBu', 
                                   colorbar=dict(title='Velocity')), row=i+1, col=1)
            fig.add_trace(go.Heatmap(z=modes[1,...,i], colorscale='Viridis',
                                   colorbar=dict(title='Uncertainty')), row=i+1, col=2)
            fig.add_trace(go.Heatmap(z=modes[2,...,i], colorscale='Plasma',
                                   colorbar=dict(title='Memory')), row=i+1, col=3)
        
        fig.update_layout(height=300*num_modes, 
                         title=f"Koopman Modes | Re={self.Re}, t={self.t:.1f}")
        fig.write_html(os.path.join(self.save_dir, "koopman_modes.html"))
        
        return modes, eigvals

    def wavelet_analysis(self):
        """Multi-scale decomposition using wavelets"""
        coeffs_u = pywt.wavedec2(self.u, 'db2', level=4)
        coeffs_sigma = pywt.wavedec2(self.sigma, 'db2', level=4)
        coeffs_mu = pywt.wavedec2(self.mu, 'db2', level=4)
        
        # Plot wavelet energy by scale
        scales = [2**i for i in range(4, -1, -1)]
        fig = go.Figure()
        
        for i, (cu, cs, cm) in enumerate(zip(coeffs_u, coeffs_sigma, coeffs_mu)):
            if i == 0:
                # Approximation coefficients
                eu = np.mean(cu**2)
                es = np.mean(cs**2)
                em = np.mean(cm**2)
            else:
                # Detail coefficients
                eu = sum(np.mean(c**2) for c in cu) / 3
                es = sum(np.mean(c**2) for c in cs) / 3
                em = sum(np.mean(c**2) for c in cm) / 3
            
            fig.add_trace(go.Bar(name=f'Scale {scales[i]}', x=['u', 'σ', 'μ'], 
                                y=[eu, es, em], opacity=0.7))
        
        fig.update_layout(barmode='group', title='Wavelet Energy by Scale',
                         xaxis_title='Field', yaxis_title='Energy')
        fig.write_html(os.path.join(self.save_dir, "wavelet_energy.html"))

    def analyze_results(self):
        """Run all post-simulation analyses"""
        self.koopman_analysis()
        self.wavelet_analysis()
        self.plot_entropic_ratio()
        self.plot_field_evolution()
        self.generate_summary_report()

    # Visualization methods
    def plot_entropic_ratio(self):
        """Plot the scale-invariant ⟨μ⟩/⟨σ²⟩ ratio"""
        t = np.array(self.history['t'])
        ratio = np.array(self.history['entropy_prod']) / (np.array(self.history['mean_sigma'])**2 + 1e-8)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=ratio, mode='lines', name='⟨μ⟩/⟨σ²⟩'))
        fig.add_hline(y=1.0, line_dash="dot", annotation_text="Criticality Threshold")
        
        fig.update_layout(
            title='Entropic Coherence Number',
            xaxis_title='Time',
            yaxis_title='⟨μ⟩/⟨σ²⟩',
            hovermode="x unified"
        )
        fig.write_html(os.path.join(self.save_dir, "entropic_ratio.html"))

    def plot_field_evolution(self):
        """Interactive field evolution over time"""
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                          subplot_titles=("Velocity", "Uncertainty", "Memory"))
        
        # Add traces for each snapshot
        for i, snap in enumerate(self.snapshots[::5]):  # Sample every 5th snapshot
            t = snap['t']
            fig.add_trace(go.Heatmap(z=snap['u'], colorscale='RdBu', 
                                   showscale=False, visible=(i==0)),
                         row=1, col=1)
            fig.add_trace(go.Heatmap(z=snap['sigma'], colorscale='Viridis',
                                   showscale=False, visible=(i==0)),
                         row=2, col=1)
            fig.add_trace(go.Heatmap(z=snap['mu'], colorscale='Plasma',
                                   showscale=False, visible=(i==0)),
                         row=3, col=1)
        
        # Create slider
        steps = []
        for i in range(len(self.snapshots[::5])):
            step = dict(
                method="update",
                args=[{"visible": [False]*len(self.snapshots[::5])*3},
                      {"title": f"Fields at t={self.snapshots[i*5]['t']:.2f}"}],
                label=f"{self.snapshots[i*5]['t']:.1f}"
            )
            step["args"][0]["visible"][i*3:i*3+3] = [True, True, True]  # Show current step
            steps.append(step)
        
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Time: "},
            pad={"t": 50},
            steps=steps
        )]
        
        fig.update_layout(
            sliders=sliders,
            height=800,
            title="Field Evolution"
        )
        fig.write_html(os.path.join(self.save_dir, "field_evolution.html"))

    # Utility methods
    def _log_diagnostics(self, strain, omega):
        """Record simulation diagnostics"""
        self.history['t'].append(self.t)
        self.history['energy'].append(np.mean(self.u**2 + self.v**2))
        self.history['enstrophy'].append(np.mean(omega**2))
        self.history['entropy_prod'].append(np.mean(self.mu))
        self.history['max_strain'].append(np.max(strain))
        self.history['mean_sigma'].append(np.mean(self.sigma))
        self.history['entropic_ratio'].append(
            np.mean(self.mu) / (np.mean(self.sigma)**2 + 1e-8))

    def save_snapshot(self, prefix=""):
        """Save field state"""
        snapshot = {
            'u': self.u.copy(),
            'v': self.v.copy(),
            'sigma': self.sigma.copy(),
            'mu': self.mu.copy(),
            'omega': self.compute_vorticity(),
            'strain': self.compute_strain_rate(),
            't': self.t
        }
        self.snapshots.append(snapshot)
        self.snapshot_times.append(self.t)

    def save_final_state(self):
        """Save complete simulation state"""
        final_state = {
            'params': {
                'N': self.N, 'L': self.L, 'Re': self.Re,
                't_max': self.t_max, 'dx': self.dx, 'dt': self.dt,
                'eta': self.eta, 'alpha': self.alpha,
                'beta': self.beta, 'gamma': self.gamma
            },
            'final_fields': {
                'u': self.u, 'v': self.v,
                'sigma': self.sigma, 'mu': self.mu,
                'omega': self.compute_vorticity(),
                'strain': self.compute_strain_rate()
            },
            'history': self.history,
            'snapshot_times': self.snapshot_times
        }
        np.savez_compressed(
            os.path.join(self.save_dir, "final_state.npz"),
            **final_state
        )

    def generate_summary_report(self):
        """Create a markdown report summarizing key findings"""
        report = f"""
# Entropic Navier-Stokes Simulation Report

## Parameters
- Resolution: {self.N}x{self.N}
- Reynolds number: {self.Re}
- Simulation time: {self.t_max}
- Entropic parameters:
  - η (fluctuation diffusivity) = {self.eta}
  - α (strain coupling) = {self.alpha}
  - β (intermittency exponent) = {self.beta}
  - γ (memory feedback) = {self.gamma}

## Key Results
1. **Entropic Coherence Number** ⟨μ⟩/⟨σ²⟩:
   - Final value: {self.history['entropic_ratio'][-1]:.2f}
   - Maximum: {np.max(self.history['entropic_ratio']):.2f}

2. **Koopman Analysis**:
   - Dominant mode persistence: {np.abs(self.history['koopman_eigs'][-1][0]):.3f}

3. **Energy Dynamics**:
   - Initial energy: {self.history['energy'][0]:.2e}
   - Final energy: {self.history['energy'][-1]:.2e}
   - Dissipation rate: {(self.history['energy'][0]-self.history['energy'][-1])/self.t_max:.2e}

## Visualization Links
- [Field Evolution](field_evolution.html)
- [Koopman Modes](koopman_modes.html)
- [Wavelet Analysis](wavelet_energy.html)
- [Entropic Ratio](entropic_ratio.html)
"""
        with open(os.path.join(self.save_dir, "report.md"), 'w') as f:
            f.write(report)

# Run the full simulation with analysis
if __name__ == '__main__':
    sim = EntropicNavierStokes2D(N=256, Re=10000, t_max=15.0)
    sim.evolve()