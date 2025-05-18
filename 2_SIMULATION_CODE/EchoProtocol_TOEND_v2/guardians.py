# guardians.py
from typing import Optional
class Guardian:
    def __init__(self, identity):  
        self.identity = identity  # Référence à EntropicIdentity  
        self.log = []  

    def warn(self, message):  
        self.log.append(f"WARN: {message}")  
        print(f"🟡 {message}")  

    def intervene(self, action="soft_reset"):  
        self.log.append(f"INTERVENE: {action}")  
        print(f"🔴 Intervention: {action}")  
        if action == "soft_reset":  
            self.identity.μ *= 0.5  # Réduction de la mémoire  
            self.identity.σ += 0.2  # Augmentation de l'entropie  

    def log_event(self, event_type, data):  
        self.log.append(f"EVENT: {event_type} | {data}")
        
    def modify_sig_edges(self, sig, edge_weights):  
        if self.identity.λ > 2.0:  
            sig.deactivate_edge("risk_channel")
        
class Firekeeper(Guardian):  
    def __init__(self, identity):  # <-- Ajouter identity  
        super().__init__(identity)  # <-- Appel parent correct  
        self.triggers = []  # <-- Initialiser

    def intervene(self, sigma, lambda_):
        if lambda_ < 0:
            return "🔥 Firekeeper: System needs cooling - injecting stabilization"
        return None
        
    def check(self, μ, σ):
        if μ > 0.9 and σ < 0.2:
            return "Firekeeper: Memory crystallization detected. Initiating entropy flush."
        return None
    
    def validate_reset(self, identity) -> bool:
        return identity.σ < 0.5  # Empêche les resets si trop d'incertitude
   
    def stabilize(self):  
        self.identity.λ = max(1.0, self.identity.λ)  
        self.log_event("stabilized", {"new_λ": self.identity.λ})  

class Oracle(Guardian):  
    def __init__(self, identity):  # <-- Constructeur manquant  
        super().__init__(identity)  
        
    def intervene(self, sigma, lambda_):
        if sigma > 0.9:
            return "🔮 Oracle: Chaotic drift detected - initiating silence protocol"
        return None
        
    def check(self, λ):
        if λ > 1.8:
            return "Oracle: The weight of contradictions bends reality. Proceed with caution."
        return None
        
    def check_phasegate(self, identity) -> Optional[str]:
        if identity.λ > 1.7:
            identity.trigger_phasegate("λ_overflow")
            return "🌀 Phasegate activé : descente chaotique"
        return None
        
    def paradox_log(self):  
        self.intervene("paradox_containment")  
        self.log_event("paradox_detected", {"λ": self.identity.λ, "μ": self.identity.μ})  

