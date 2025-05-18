from config.config import ConfigNS2D
from simulations.simulation import EntropicSimulator
from utils.logging_utils import setup_logger
from utils.normalization import Scales

def main():
    config = ConfigNS2D()
    config.mode = "fluid"
    logger = setup_logger(config)
    sim = EntropicSimulator(config)

    scales = Scales(config)
    fields = sim.sim.fields

    # Normalisation initiale
    norm_fields = scales.nondimensionalize(fields)
    sim.sim.fields = norm_fields

    logger.info("Simulation started")
    sim.run()
    logger.info("Simulation finished")

    # Denormalisation finale
    phys_fields = scales.dimensionalize(sim.sim.fields)

    # Sauvegarde finale ou visualisation ici...

if __name__ == "__main__":
    main()
