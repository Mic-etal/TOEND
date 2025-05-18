# utils/visualization.py
import os
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Visualizer:
    def __init__(self, config):
        """
        Classe de visualisation pour les simulations TOEND NS2D.
        
        Args:
            config: Objet configuration contenant notamment save_path, use_gpu
        """
        self.config = config
        os.makedirs(config.save_path, exist_ok=True)

    def _to_cpu(self, array):
        """
        Convertit un tableau CuPy en NumPy si nécessaire.
        """
        if cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array

    def save_snapshot(self, fields, suffix=""):
        """
        Sauvegarde un snapshot compressé en npz.
        
        Args:
            fields (dict): dictionnaire des champs à sauvegarder
            suffix (str): suffixe optionnel pour le nom du fichier
        """
        snapshot = {k: self._to_cpu(v) for k, v in fields.items()}
        filename = os.path.join(
            self.config.save_path,
            f"snapshot_t{fields['t']:.2f}{suffix}.npz"
        )
        np.savez_compressed(filename, **snapshot)

    def plot_diagnostics(self, diagnostics):
        """
        Trace les diagnostics clés avec style professionnel.
        
        Args:
            diagnostics (dict): dictionnaire contenant les séries temporelles
        """
        plt.style.use('seaborn-v0_8-poster')
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))

        self._plot_timeseries(axs[0, 0], diagnostics.get('time', []), diagnostics.get('energy', []),
                             'Energy', r'$\frac{1}{2}\langle u^2 + v^2 \rangle$')
        self._plot_timeseries(axs[0, 1], diagnostics.get('time', []), diagnostics.get('enstrophy', []),
                             'Enstrophy', r'$\langle \omega^2 \rangle$')
        self._plot_timeseries(axs[0, 2], diagnostics.get('time', []), diagnostics.get('mean_nstar', []),
                             'Mean n*', r'$\langle n^* \rangle$')

        self._plot_timeseries(axs[1, 0], diagnostics.get('time', []), diagnostics.get('mean_sigma', []),
                             'Mean σ', r'$\langle \sigma \rangle$', color='darkorange')
        self._plot_timeseries(axs[1, 1], diagnostics.get('time', []), diagnostics.get('mean_mu', []),
                             'Mean μ', r'$\langle \mu \rangle$', color='purple')

        axs[1, 2].semilogy(diagnostics.get('time', [])[1:],
                          np.diff(diagnostics.get('time', [1])),
                          'k-')
        axs[1, 2].set_title('Adaptive Timestep')
        axs[1, 2].set_xlabel('Time')
        axs[1, 2].set_ylabel('Δt')
        axs[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.save_path, 'diagnostics.png'), dpi=150)
        plt.close()

    def _plot_timeseries(self, ax, x, y, title, ylabel, color='steelblue'):
        if len(x) == 0 or len(y) == 0:
            ax.text(0.5, 0.5, 'No data to display',
                    ha='center', va='center',
                    transform=ax.transAxes)
        else:
            ax.plot(x, y, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    def create_field_animation(self, snapshots, field='vorticity', fps=15):
        """
        Génère une animation MP4 de l'évolution d'un champ.
        
        Args:
            snapshots (list): liste de fichiers npz contenant les snapshots
            field (str): nom du champ à animer
            fps (int): images par seconde
        Returns:
            str: chemin du fichier vidéo généré
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')

        # Chargement snapshot initial
        data = np.load(snapshots[0])
        X, Y = data['X'], data['Y']
        field_data = self._get_field_data(data, field)

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
        """
        Extrait les champs dérivés ou directs selon le champ demandé.
        """
        if field_name == 'vorticity':
            return data['vx'] - data['uy']  # ∂v/∂x - ∂u/∂y
        elif field_name == 'strain':
            return np.sqrt(2*(data['ux']**2 + data['vy']**2) + (data['uy'] + data['vx'])**2)
        else:
            return data[field_name]

    def interactive_dashboard(self, diagnostics):
        """
        Lance un dashboard interactif Bokeh (Jupyter compatible).
        Fallback sur matplotlib si Bokeh absent.
        """
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


# Ajout d’une docstring globale du module

"""
Ce module fournit une classe Visualizer dédiée à la visualisation des simulations TOEND NS2D.
Il supporte l’enregistrement des snapshots, le traçage de diagnostics, la génération d’animations,
ainsi que des dashboards interactifs avec Bokeh.
La gestion GPU/CPU est assurée automatiquement via la conversion CuPy/NumPy.
"""

