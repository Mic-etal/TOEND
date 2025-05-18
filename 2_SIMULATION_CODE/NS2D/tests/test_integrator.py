import numpy as np
import pytest
from config.config import ConfigNS2D
from core.operators import Operators
from core.physics import PhysicsCore
from time_integrator import EntropicNS2DIntegrator

@pytest.fixture
def config():
    return ConfigNS2D()

@pytest.fixture
def setup_simulation(config):
    grid = config.create_grid()
    ops = Operators(grid)
    phys = PhysicsCore(config, ops)
    integrator = EntropicNS2DIntegrator(config, phys, ops)
    shape = (config.N, config.N)
    fields = {
        'u': np.zeros(shape),
        'v': np.zeros(shape),
        'sigma': np.ones(shape) * 0.1,
        'mu': np.zeros(shape),
        'n_star': np.ones(shape) * 2.0,
        't': 0.0
    }
    return integrator, fields

def test_integrator_run_short(setup_simulation):
    integrator, fields = setup_simulation
    integrator.run(fields, steps=10)
    
    # Vérifier que le temps a bien avancé
    assert fields['t'] > 0
    
    # Vérifier qu'il n'y a pas de NaN ni inf dans les champs
    for key, val in fields.items():
        if isinstance(val, np.ndarray):
            assert np.all(np.isfinite(val)), f"Champ {key} contient NaN ou inf"
        else:
            # t est un float
            assert np.isfinite(val), f"Temps t contient NaN ou inf"
