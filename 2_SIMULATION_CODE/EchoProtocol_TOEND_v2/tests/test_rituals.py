from core import EntropicIdentity  
from guardian import Firekeeper, Oracle  
import rituals  

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