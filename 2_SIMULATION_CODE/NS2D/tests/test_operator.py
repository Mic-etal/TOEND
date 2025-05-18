import numpy as np
import pytest
from config.config import ConfigNS2D
from core.operators import Operators

@pytest.fixture
def config():
    return ConfigNS2D()

@pytest.fixture
def setup_operators(config):
    grid = config.create_grid()
    ops = Operators(grid)
    return ops

def test_gradient_shape_and_type(setup_operators):
    ops = setup_operators
    f = np.random.rand(ops.grid.kx.shape[0], ops.grid.kx.shape[1])
    fx, fy = ops.grad(f)
    assert fx.shape == f.shape
    assert fy.shape == f.shape
    assert isinstance(fx, np.ndarray)
    assert isinstance(fy, np.ndarray)

def test_gradient_of_constant(setup_operators):
    ops = setup_operators
    f = np.ones_like(ops.grid.kx)
    fx, fy = ops.grad(f)
    # Gradient d'une constante doit être proche de zéro
    assert np.allclose(fx, 0, atol=1e-10)
    assert np.allclose(fy, 0, atol=1e-10)

def test_laplacian_shape_and_type(setup_operators):
    ops = setup_operators
    f = np.random.rand(ops.grid.kx.shape[0], ops.grid.kx.shape[1])
    lap = ops.laplacian(f)
    assert lap.shape == f.shape
    assert isinstance(lap, np.ndarray)

def test_laplacian_of_constant(setup_operators):
    ops = setup_operators
    f = np.ones_like(ops.grid.kx)
    lap = ops.laplacian(f)
    # Laplacien d'une constante doit être proche de zéro
    assert np.allclose(lap, 0, atol=1e-10)
