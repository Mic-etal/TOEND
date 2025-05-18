# visualization.py
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from scipy.ndimage import gaussian_filter
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def plot_diagnostics(self, diagnostics):
        """Crash-resistant plotting"""
        try:
            # Conversion sécurisée en array numpy
            time = np.array(diagnostics.get('time', []))
            energy = np.nan_to_num(diagnostics.get('energy', []), nan=0, posinf=0, neginf=0)
            
            # Création des figures avec vérification des données
            if len(time) == 0:
                self.plot_empty("No valid time points recorded")
                return
                
            # ... reste du code de visualisation ...
            
        except Exception as e:
            self.plot_empty(f"Plotting failed: {str(e)}")

    def plot_empty(self, message):
        """Display empty plot with error message"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message,
               ha='center', va='center',
               transform=ax.transAxes)
        ax.set_title("Simulation Diagnostics - Error")
        plt.savefig(os.path.join(self.config.save_path, 'diagnostics_error.png'))
        plt.close()
        
    def __init__(self, config):
        self.config = config
        os.makedirs(config.save_path, exist_ok=True)

    def save_snapshot(self, fields, suffix=""):
        """Save field snapshots as compressed NPZ files"""
        if self.config.use_gpu:
            snapshot = {k: cp.asnumpy(v) for k, v in fields.items()}
        else:
            snapshot = fields.copy()

        filename = os.path.join(
            self.config.save_path,
            f"snapshot_t{fields['t']:.2f}{suffix}.npz"
        )
        np.savez_compressed(filename, **snapshot)

    def plot_diagnostics(self, diagnostics):
        """Generate diagnostic plots with professional styling"""
        plt.style.use('seaborn-v0_8-poster')
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))

        # Time series plots
        self._plot_timeseries(axs[0, 0], diagnostics['time'], diagnostics['energy'],
                             'Energy', r'$\frac{1}{2}\langle u^2 + v^2 \rangle$')
        self._plot_timeseries(axs[0, 1], diagnostics['time'], diagnostics['enstrophy'],
                             'Enstrophy', r'$\langle \omega^2 \rangle$')
        self._plot_timeseries(axs[0, 2], diagnostics['time'], diagnostics['mean_nstar'],
                             'Mean n*', r'$\langle n^* \rangle$')
        
        # Field statistics
        self._plot_timeseries(axs[1, 0], diagnostics['time'], diagnostics['mean_sigma'],
                             'Mean σ', r'$\langle \sigma \rangle$', color='darkorange')
        self._plot_timeseries(axs[1, 1], diagnostics['time'], diagnostics['mean_mu'],
                             'Mean μ', r'$\langle \mu \rangle$', color='purple')
        
        # Timestep evolution
        axs[1, 2].semilogy(diagnostics['time'][1:], 
                          np.diff(diagnostics['time']), 'k-')
        axs[1, 2].set_title('Adaptive Timestep')
        axs[1, 2].set_xlabel('Time')
        axs[1, 2].set_ylabel('Δt')
        axs[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.save_path, 'diagnostics.png'), dpi=150)
        plt.close()

    def _plot_timeseries(self, ax, x, y, title, ylabel, color='steelblue'):
        ax.plot(x, y, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    def create_field_animation(self, snapshots, field='vorticity', fps=15):
        """Generate MP4 animation of field evolution"""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')
        
        # Load first snapshot to initialize
        data = np.load(snapshots[0])
        X, Y = data['X'], data['Y']
        field_data = self._get_field_data(data, field)
        
        # Create initial plot
        vmin, vmax = np.percentile(field_data, [5, 95])
        im = ax.pcolormesh(X, Y, field_data, 
                          shading='auto', cmap='viridis',
                          vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=field)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                           color='white', fontsize=12)

        def update(frame):
            data = np.load(frame)
            field_data = self._get_field_data(data, field)
            im.set_array(field_data.ravel())
            time_text.set_text(f"t = {data['t']:.2f}")
            return im, time_text

        ani = FuncAnimation(fig, update, frames=snapshots,
                           interval=1000/fps, blit=True)
        
        output_path = os.path.join(self.config.save_path, f'{field}_evolution.mp4')
        ani.save(output_path, writer='ffmpeg', fps=fps, dpi=200)
        plt.close()
        return output_path

    def _get_field_data(self, data, field_name):
        """Extract derived fields from snapshot data"""
        if field_name == 'vorticity':
            return data['vx'] - data['uy']  # ∂v/∂x - ∂u/∂y
        elif field_name == 'strain':
            return np.sqrt(2*(data['ux']**2 + data['vy']**2) + (data['uy'] + data['vx'])**2)
        else:
            return data[field_name]

    def interactive_dashboard(self, diagnostics):
        """Launch interactive Bokeh dashboard (Jupyter compatible)"""
        try:
            from bokeh.plotting import figure, output_notebook, show
            from bokeh.layouts import gridplot
            output_notebook()
            
            tools = "pan,wheel_zoom,box_zoom,reset,save"
            p1 = figure(title="Energy", tools=tools, width=400, height=300)
            p1.line(diagnostics['time'], diagnostics['energy'], line_width=2)
            
            p2 = figure(title="Enstrophy", tools=tools, width=400, height=300)
            p2.line(diagnostics['time'], diagnostics['enstrophy'], color="red", line_width=2)
            
            p3 = figure(title="Mean Fields", tools=tools, width=400, height=300)
            p3.line(diagnostics['time'], diagnostics['mean_sigma'], legend_label="σ", line_width=2)
            p3.line(diagnostics['time'], diagnostics['mean_mu'], legend_label="μ", color="green", line_width=2)
            
            grid = gridplot([[p1, p2], [p3, None]])
            show(grid)
        except ImportError:
            print("Bokeh not available - falling back to matplotlib")
            self.plot_diagnostics(diagnostics)
            # visualization.py - Extensions
def plot_dashboard(fields, grid, t):
    """Affiche les 6 champs clés avec projections spectrales."""
    fig = plt.figure(figsize=(18, 12))
    
    # Configuration des plots
    plots = [
        ('u', 'RdBu', 'Vitesse X'),
        ('v', 'RdBu', 'Vitesse Y'), 
        ('sigma', 'viridis', 'Entropie σ'),
        ('mu', 'plasma', 'Mémoire μ'),
        ('n_star', 'coolwarm', 'Dimension n*'),
        ('omega', 'PRGn', 'Vorticité ω')
    ]
    
    for i, (field, cmap, title) in enumerate(plots, 1):
        ax = fig.add_subplot(2, 3, i)
        im = ax.imshow(fields[field], cmap=cmap)
        plt.colorbar(im, ax=ax, label=title)
        ax.set_title(f"{title} à t={t:.2f}")
    
    plt.tight_layout()
    return fig
    
def plot_snapshot(x, sigma, mu, time):
    plt.figure(figsize=(10,5))
    plt.plot(x, sigma, 'r-', label=f'σ (t={time:.2f})')
    plt.plot(x, mu, 'b-', label='μ')
    plt.xlabel('Position'); plt.ylabel('Value')
    plt.legend()
    plt.title(f'σ-μ System at t={time:.2f}')
    plt.show()

def animate_snapshots(snapshot_files):
    """Génère une animation MP4/HTML à partir des snapshots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    anim = FuncAnimation(fig, update_frame, frames=snapshot_files, fargs=(axes,))
    anim.save("simulation.mp4", writer='ffmpeg', dpi=150)
    return HTML(anim.to_html5_video())
    


def plot_phase(sigma, mu, save=False):
    plt.figure(figsize=(10,4))
    
    plt.subplot(121)
    plt.plot(sigma, mu, 'k.', alpha=0.5)
    plt.xlabel("σ"); plt.ylabel("μ")
    plt.title("Espace des Phases")
    
    plt.subplot(122)
    plt.plot(sigma, label="σ")
    plt.plot(mu, label="μ")
    plt.legend(); plt.title("Profils Spatiaux")
    
    if save:
        plt.savefig(f"phase_{int(time.time())}.png")
    plt.close()
    
def plot_history(history):
    fig, axs = plt.subplots(2, 2, figsize=(12,8))
    
    # Évolution temporelle
    axs[0,0].plot(history['time'], history['integral_sigma'])
    axs[0,0].set_title('∫σ over time')
    
    axs[0,1].plot(history['time'], history['max_mu'])
    axs[0,1].set_title('max(μ) over time')
    
    # Portrait de phase
    axs[1,0].scatter(history['integral_sigma'], history['max_mu'], 
                    c=history['time'], cmap='viridis')
    axs[1,0].set_xlabel('∫σ'); axs[1,0].set_ylabel('max(μ)')
    
    # Distribution finale
    axs[1,1].plot(x, history['sigma'][-1], 'r-', label='σ')
    axs[1,1].plot(x, history['mu'][-1], 'b-', label='μ')
    axs[1,1].legend()
    
    import matplotlib.pyplot as plt

def plot_field_evolution(x, sigma, mu, current_time):
    """Visualisation temporelle"""
    plt.figure(figsize=(12,5))
    plt.plot(x, sigma, 'r-', label='σ (entropie)')
    plt.plot(x, mu, 'b-', label='μ (mémoire)')
    plt.xlabel('Position'); plt.ylabel('Valeur')
    plt.title(f'Évolution à t = {current_time:.2f}')
    plt.legend()
    plt.show()

def plot_phase_portrait(sigma_history, mu_history):
    """Portrait de phase final"""
    plt.figure(figsize=(8,8))
    for s, m in zip(sigma_history, mu_history):
        plt.scatter(s, m, alpha=0.1, c='k')
    plt.xlabel('σ'); plt.ylabel('μ')
    plt.title('Portrait de phase σ-μ')
    plt.grid(True)
    plt.show()