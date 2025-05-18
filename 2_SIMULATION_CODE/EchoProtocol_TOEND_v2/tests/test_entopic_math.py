import pytest
from core import EntropicIdentity, EntropicMath

def test_entropic_add_non_associativity():
    a = EntropicIdentity(μ=0.5, σ=0.4)
    b = EntropicIdentity(μ=0.3, σ=0.6)
    c = EntropicIdentity(μ=0.7, σ=0.2)
    
    left = EntropicMath.add(EntropicMath.add(a, b), c)
    right = EntropicMath.add(a, EntropicMath.add(b, c))
    
    assert left.μ != right.μ, "μ doit diverger"
    assert left.σ != right.σ, "σ doit diverger"
    assert abs(left.λ - right.λ) > 0.1, "λ doit montrer une bifurcation"