# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.ndimage import gaussian_filter

try:
    import cupy as cp
except ImportError:
    cp = None

class Visualizer:
    """A class for creating and managing simulation visualizations."""
    
    def __init__(self, config):
        self.config = config
        self.style = {
            'figure.figsize': (12, 8),
            'font.size': 12,
            'lines.linewidth': 2,
            'axes.grid': True,
            'grid.alpha': 0.3
        }
        os.makedirs(config.save_path, exist_ok=True)

    def _apply_style(self):
        """Apply consistent plotting style."""
        plt.style.use('seaborn-v0_8-poster')
        plt.rcParams.update(self.style)

    def plot_diagnostics(self, diagnostics):
        """Generate diagnostic plots with error handling."""
        try:
            self._apply_style()
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            
            # Validate data
            time = np.array(diagnostics.get('time', []))
            if len(time) == 0:
                raise ValueError("No valid time points recorded")

            # Time series plots
            self._plot_timeseries(axs[0, 0], time, diagnostics['energy'],
                                 'Energy', r'$\frac{1}{2}\langle u^2 + v^2 \rangle$')
            self._plot_timeseries(axs[0, 1], time, diagnostics['enstrophy'],
                                 'Enstrophy', r'$\langle \omega^2 \rangle$')
            self._plot_timeseries(axs[0, 2], time, diagnostics['mean_nstar'],
                                 'Mean n*', r'$\langle n^* \rangle$')
            
            # Field statistics
            self._plot_timeseries(axs[1, 0], time, diagnostics['mean_sigma'],
                                 'Mean σ', r'$\langle \sigma \rangle$', 'darkorange')
            self._plot_timeseries(axs[1, 1], time, diagnostics['mean_mu'],
                                 'Mean μ', r'$\langle \mu \rangle$', 'purple')
            
            # Timestep evolution
            self._plot_timesteps(axs[1, 2], time, np.diff(time))

            plt.tight_layout()
            plt.savefig(os.path.join(self.config.save_path, 'diagnostics.png'), dpi=150)
            plt.close()

        except Exception as e:
            self._plot_error(f"Diagnostics plot failed: {str(e)}")

    def _plot_timeseries(self, ax, x, y, title, ylabel, color='steelblue'):
        """Helper method for creating time series plots."""
        ax.plot(x, y, color=color)
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    def _plot_timesteps(self, ax, time, dt):
        """Plot adaptive timestep evolution."""
        ax.semilogy(time[1:], dt, 'k-')
        ax.set_title('Adaptive Timestep')
        ax.set_xlabel('Time')
        ax.set_ylabel('Δt')
        ax.grid(True)

    def create_field_animation(self, snapshots, field='vorticity', fps=15):
        """Generate field evolution animation."""
        self._apply_style()
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')

        # Initialize plot
        data = np.load(snapshots[0])
        X, Y = data['X'], data['Y']
        field_data = self._get_field_data(data, field)

        vmin, vmax = np.percentile(field_data, [5, 95])
        im = ax.pcolormesh(X, Y, field_data, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=field)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)

        def update(frame):
            data = np.load(frame)
            field_data = self._get_field_data(data, field)
            im.set_array(field_data.ravel())
            time_text.set_text(f"t = {data['t']:.2f}")
            return im, time_text

        ani = FuncAnimation(fig, update, frames=snapshots, interval=1000//fps, blit=True)
        output_path = os.path.join(self.config.save_path, f'{field}_evolution.mp4')
        ani.save(output_path, writer='ffmpeg', fps=fps, dpi=200)
        plt.close()
        return output_path

    def _get_field_data(self, data, field_name):
        """Calculate derived fields from raw data."""
        field_map = {
            'vorticity': data['vx'] - data['uy'],
            'strain': np.sqrt(2*(data['ux']**2 + data['vy']**2) + (data['uy'] + data['vx'])**2),
        }
        return field_map.get(field_name, data[field_name])

    def _plot_error(self, message):
        """Display error message in empty plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Visualization Error")
        plt.savefig(os.path.join(self.config.save_path, 'error.png'))
        plt.close()

    def save_snapshot(self, fields, suffix=""):
        """Save field snapshots as compressed NPZ files."""
        snapshot = {k: cp.asnumpy(v) if self.config.use_gpu and cp else v.copy() 
                   for k, v in fields.items()}
        filename = os.path.join(
            self.config.save_path,
            f"snapshot_t{fields['t']:.2f}{suffix}.npz"
        )
        np.savez_compressed(filename, **snapshot)

    def plot_phase_space(self, sigma_history, mu_history):
        """Create phase space visualization."""
        self._apply_style()
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].scatter(sigma_history, mu_history, alpha=0.1, c='k')
        ax[0].set_xlabel('σ'); ax[0].set_ylabel('μ')
        ax[0].set_title('Phase Space')

        ax[1].loglog(sigma_history, mu_history, 'o', alpha=0.6)
        ax[1].set_xlabel('σ (bits)'); ax[1].set_ylabel('μ (seconds)')
        ax[1].set_title('Scaling Law')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.save_path, 'phase_space.png'))
        plt.close()

    def plot_field_comparison(self, x, sigma, mu, current_time):
        """Plot field distributions at current time."""
        self._apply_style()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, sigma, 'r-', label='Entropy (σ)')
        ax.plot(x, mu, 'b-', label='Memory (μ)')
        ax.set_xlabel('Position'); ax.set_ylabel('Value')
        ax.set_title(f'Field Distribution at t = {current_time:.2f}')
        ax.legend()
        plt.savefig(os.path.join(self.config.save_path, f'fields_t{current_time:.2f}.png'))
        plt.close()
        
    def plot_triplet_evolution(history):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(history['time'], history['mean_sigma'], label='σ')
        ax[0].plot(history['time'], history['mean_mu'], label='μ')
        ax[0].set_title('Entropy-Memory Dynamics')
    
        ax[1].loglog(history['mean_sigma'], history['mean_mu'], 'o-')
        ax[1].set_title('σ-μ Scaling')
    
        ax[2].plot(history['time'], history['x_variability'])
        ax[2].set_title('Observable (x) Variability')
        
        
    def plot_phase_portrait(data, title="Phase Portrait"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Example plotting code
        ax.scatter(data[:,0], data[:,1], data[:,2])
        ax.set_title(title)
        plt.show()

    def plot_field_evolution(field, timesteps):
        # Implementation for field evolution plot
        pass