import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_data(path='data/simulation_output.json'):
    """Charge les données de simulation depuis un fichier JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier {path} est introuvable.")
    with open(path, 'r') as f:
        return json.load(f)

def plot_trajectories(data):
    t = np.array(data['time'])
    mu = np.array(data['mu'])
    sigma = np.array(data['sigma'])
    lam = np.array(data['lambda'])

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(t, mu, label='μ(t)', color='royalblue')
    axs[0].set_ylabel('μ (Mémoire)')
    axs[0].legend()

    axs[1].plot(t, sigma, label='σ(t)', color='darkorange')
    axs[1].set_ylabel('σ (Incertitude)')
    axs[1].legend()

    axs[2].plot(t, lam, label='λ(t)', color='forestgreen')
    axs[2].set_ylabel('λ (Tension)')
    axs[2].set_xlabel('Temps')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def plot_phase_space(data):
    mu = np.array(data['mu'])
    sigma = np.array(data['sigma'])

    plt.figure(figsize=(6, 6))
    plt.plot(mu, sigma, color='slateblue', alpha=0.7)
    plt.xlabel('μ')
    plt.ylabel('σ')
    plt.title('Espace des phases (μ, σ)')
    plt.grid(True)
    plt.show()

# visualize.py (new methods)
def live_plotter(data_stream):
    """Plot real-time μ/σ/λ using matplotlib animation"""
    fig, axs = plt.subplots(3, 1)
    def animate(i):
        axs[0].clear()
        axs[0].plot(data_stream['mu'][-50:], color='royalblue')
        axs[0].set_ylabel('μ')
        # Repeat for σ and λ
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()

if __name__ == "__main__":
    data = load_data()
    plot_trajectories(data)
    plot_phase_space(data)
