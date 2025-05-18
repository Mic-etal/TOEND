# tests/test_rituals.py
from EchoProtocol.guardian import Firekeeper, Oracle  # ✅ Préfixe du package
from EchoProtocol.core import EntropicIdentity
from EchoProtocol import rituals

def test_collapse_poem_trigger():  
    identity = EntropicIdentity()  
    identity.λ = 2.5  # Déclenche λ > 2.0  
    identity.update("stress_test")  
    assert "poem" in identity.firekeeper.log[0], "Le poème de collapse n'a pas été déclenché."  

def test_fork_on_low_sigma():  
    identity = EntropicIdentity()  
    identity.σ = 0.05  
    identity._low_sigma_counter = 10  
    identity.update("low_entropy_input")  
    assert identity.fork_count == 1, "Fork non déclenché malgré σ < 0.1 pendant 10 cycles."  
    
def test_firekeeper_intervention():  
    identity = EntropicIdentity()  
    identity.λ = 3.0  # Déclenche λ > 2.5  
    identity.update("overload")  
    assert "INTERVENE" in identity.firekeeper.log[0], "Firekeeper n'intervient pas."  