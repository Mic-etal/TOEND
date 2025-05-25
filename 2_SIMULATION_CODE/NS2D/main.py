import argparse
import sys
import os
from core.fields import FieldContainer
from utils.visualization import Visualizer
from config.config import ConfigNS2D
from simulations.simulation import EntropicSimulator

# Unicode output safe
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def main():
    parser = argparse.ArgumentParser(
        description="TOEND NS2D: Entropic Simulator (fluid & sigma-mu unified)"
    )
    parser.add_argument('--mode', type=str, default="fluid", choices=["fluid", "sigma_mu"],
                        help='Simulation mode: "fluid" (2D NS) or "sigma_mu" (1D)')
    # Placeholder for YAML/JSON config (inactive for now)
    # parser.add_argument('--config', type=str, default=None,
    #                     help='(Optional) Path to a YAML/JSON config file (not implemented)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Override save path (optional)')
    args = parser.parse_args()

    # --- CONFIG SETUP (simple: Python only) ---
    config = ConfigNS2D()
    config.mode = args.mode
    viz = Visualizer(config)
    
    # Optionally override save_path
    if args.save_path:
        config.save_path = args.save_path
        os.makedirs(config.save_path, exist_ok=True)

    print(f">> Running TOEND Simulation | mode={config.mode}")
    sim = EntropicSimulator(config)
    sim.run()
    
    fields = FieldContainer((256, 256), config, use_gpu=True)
    Nsteps = int(config.tmax / config.dt)

    for step in range(Nsteps):
        # Update NS2D (u, v, sigma, mu, etc.)
        # update_ns2d(fields.u, fields.v, fields.sigma, ...)  # À adapter

        # Update n*
        fields.evolve_n_star(dt=0.01)

        # Diagnostics
        stats = fields.get_scalar_stats()
        print(f"Step {step}: S={stats['S']:.2f}, n*={stats['n_star']:.2f}")

        # Live viz (tous les 10 pas)
        if step % 10 == 0:
            Visualizer.live_plot(fields, step, show="n_star")

if __name__ == "__main__":
    main()
