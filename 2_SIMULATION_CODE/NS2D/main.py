import argparse
from config import mode  # default fallback
import sys

# -- Terminal compatibility fix --
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

# -- Argument parser --
parser = argparse.ArgumentParser(description="Run Epsilon simulation.")
parser.add_argument('--mode', type=str, default=mode, help='Simulation mode: fluid or sigma_mu')
args = parser.parse_args()

# -- Routing --
if args.mode == "sigma_mu":
    print(">> Running sigma-mu simulation [OK]")
    from simulation_sigma_mu import run_sigma_mu
    run_sigma_mu()
elif args.mode == "fluid":
    print(">> Running full entropic fluid simulation [OK]")
    from simulation import EntropicNS2DSimulation
    from config import SimulationConfig  # si ce n'est pas encore importé
    sim = EntropicNS2DSimulation(SimulationConfig)
    sim.run()
else:
    raise ValueError(f"Unknown mode: {args.mode}")
