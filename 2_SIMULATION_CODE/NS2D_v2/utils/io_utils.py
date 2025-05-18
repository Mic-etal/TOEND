# io_utils.py - Nouvelle version
import h5py
import numpy as np
from operators import curl, strain_rate

def save_snapshot(path, fields, grid, compress=True):
    """Sauvegarde tous les champs + dérivés avec métadonnées."""
    with h5py.File(path, 'w') as f:
        # Champs de base
        for k in ['u', 'v', 'sigma', 'mu', 'n_star']:
            f.create_dataset(k, data=fields[k], compression='gzip' if compress else None)
        
        # Dérivés calculés
        f.create_dataset('omega', data=curl(fields['u'], fields['v'], grid['kx'], grid['ky'], fields['n_star']))
        f.create_dataset('strain', data=strain_rate(fields['u'], fields['v'], grid['kx'], grid['ky'], fields['n_star']))
        
        # Métadonnées
        f.attrs['t'] = fields['t']
        f.attrs['config'] = str(grid['config'])

def load_snapshot(path, config):
    """Recharge une simulation avec vérification de cohérence."""
    with h5py.File(path, 'r') as f:
        fields = {k: f[k][:] for k in ['u', 'v', 'sigma', 'mu', 'n_star']}
        fields['t'] = f.attrs['t']
        
        # Vérification GPU/CPU
        if config['use_gpu'] and not isinstance(fields['u'], cp.ndarray):
            fields = {k: cp.asarray(v) for k, v in fields.items()}
    return fields