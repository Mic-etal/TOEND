from config.config import ConfigNS2D
from simulations.simulation import EntropicSimulator

def main():
    config = ConfigNS2D()
    config.mode = "fluid"        # mode NS2D fluide
    config.use_gpu = False       # ou True si tu as CuPy et GPU
    sim = EntropicSimulator(config)
    sim.run()
    # Optionnel : modifier quelques paramètres
    config.dt = 0.005
    config.save_path = "results/my_run"

if __name__ == "__main__":
    main()
