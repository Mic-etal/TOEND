import numpy as np
import pytest
from config.config import ConfigNS2D
from core.operators import Operators
from core.physics import PhysicsCore

@pytest.fixture
def config():
    return ConfigNS2D()

@pytest.fixture
def setup_physics(config):
    grid = config.create_grid()
    ops = Operators(grid)
    phys = PhysicsCore(config, ops)
    shape = (config.N, config.N)
    fields = {
        'u': np.zeros(shape),
        'v': np.zeros(shape),
        'sigma': np.ones(shape) * 0.1,
        'mu': np.zeros(shape),
        'n_star': np.ones(shape) * 2.0,
        't': 0.0
    }
    return phys, fields

def test_physics_step_updates(setup_physics):
    phys, fields = setup_physics
    new_fields = phys.step(fields, dt=0.01)
    # Check keys are present
    for key in ['u', 'v', 'sigma', 'mu', 'n_star', 't']:
        assert key in new_fields
    # Check no NaN or Inf
    for key, val in new_fields.items():
        if isinstance(val, np.ndarray):
            assert np.all(np.isfinite(val)), f"Field {key} contains NaN or Inf"
        else:
            assert np.isfinite(val), f"Time t contains NaN or Inf"
