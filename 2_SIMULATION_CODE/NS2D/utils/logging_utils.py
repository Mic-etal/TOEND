import os
import logging

def setup_logger(config=None, enabled=True):
    """
    Crée et configure un logger.
    - Si enabled=False, renvoie un logger Null (aucun log).
    - Si enabled=True, crée le dossier save_path si besoin et active le logger complet.
    """
    logger = logging.getLogger("simulation_logger")
    logger.handlers = []  # Réinitialise les handlers pour éviter les doublons

    if not enabled:
        logger.addHandler(logging.NullHandler())
        return logger

    save_path = getattr(config, "save_path", "results") if config is not None else "results"
    os.makedirs(save_path, exist_ok=True)

    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(f"{save_path}/simulation.log")
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
