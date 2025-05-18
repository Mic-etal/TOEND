# tests/test_core.py
import pytest
from core import EntropicIdentity

def test_lambda_bounding():
    identity = EntropicIdentity(μ_init=100.0, σ_init=0.1)
    assert identity.λ == np.tanh(1000.0)  # Verify tanh bounding

def test_stability_reset():
    identity = EntropicIdentity()
    identity.update_entropy(Δμ=10.0, Δσ=0.01)  # Force collapse
    assert identity.μ == 1.0  # Verify reset