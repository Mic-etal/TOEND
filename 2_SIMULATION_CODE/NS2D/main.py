import argparse
import sys
import os

# Unicode output safe
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from config import ConfigNS2D
from simulations.simulation import EntropicSimulator

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

    # Optionally override save_path
    if args.save_path:
        config.save_path = args.save_path
        os.makedirs(config.save_path, exist_ok=True)

    print(f">> Running TOEND Simulation | mode={config.mode}")
    sim = EntropicSimulator(config)
    sim.run()

if __name__ == "__main__":
    main()
