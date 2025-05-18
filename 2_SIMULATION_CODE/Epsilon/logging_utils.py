# logging_utils.py
import logging
from datetime import datetime

def setup_logger(config):
    logger = logging.getLogger('turbulence_sim')
    logger.setLevel(logging.INFO)
    
    # Créer un handler pour écrire dans un fichier
    fh = logging.FileHandler(f"{config.save_path}/simulation.log")
    fh.setLevel(logging.INFO)
    
    # Créer un handler pour la console
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    
    # Format des messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger