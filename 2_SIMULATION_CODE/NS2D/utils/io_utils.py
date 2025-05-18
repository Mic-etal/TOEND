# utils/io_utils.py
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
import h5py
import json
from core.operators import EntropicOperators


def save_snapshot(fields, grid, config):
    """
    Sauvegarde un snapshot compressé au format HDF5.
    Enregistre champs physiques, champs dérivés (vorticité, taux de déformation), et métadonnées.
    
    Args:
        fields (dict): dictionnaire des champs scalaires/vecteurs (u, v, sigma, mu, etc.)
        grid (dict): informations sur la grille (optionnel pour métadonnées)
        config (object): configuration de la simulation (pour sauvegarde metadata)
    """
    filename = f"{config.save_path}/snapshot_t{fields['t']:.2f}.h5"
    with h5py.File(filename, "w") as f:
        for key, val in fields.items():
            if cp is not None and isinstance(val, cp.ndarray):
                val = cp.asnumpy(val)
            f.create_dataset(key, data=val)
        
        # Champs dérivés calculés
        omega = curl(fields['u'], fields['v'])
        ops = EntropicOperators(config, grid)
        vorticity = ops.curl(u, v)
        strain = ops.strain_rate(u, v)
        if cp is not None and isinstance(omega, cp.ndarray):
            omega = cp.asnumpy(omega)
        if cp is not None and isinstance(strain, cp.ndarray):
            strain = cp.asnumpy(strain)

        f.create_dataset("omega", data=omega)
        f.create_dataset("strain", data=strain)

        # Métadonnées
        f.attrs["time"] = fields['t']
        f.attrs["config"] = json.dumps(vars(config))

def load_snapshot(filepath, use_gpu=False):
    """
    Charge un snapshot HDF5, avec option d'import GPU.
    
    Args:
        filepath (str): chemin vers le fichier .h5
        use_gpu (bool): si True, convertit les données en CuPy arrays
        
    Returns:
        data (dict): champs chargés
        t (float): temps de simulation enregistré
        config_json (str): config sérialisée en JSON
    """
    with h5py.File(filepath, "r") as f:
        data = {key: f[key][()] for key in f.keys()}
        t = f.attrs.get("time", 0.0)
        config_json = f.attrs.get("config", "{}")
    
    if use_gpu and cp is not None:
        for key in data:
            data[key] = cp.asarray(data[key])
    
    return data, t, config_json
