# -*- coding: utf-8 -*-
# core.py - Foundational TOEND mechanics
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from operator import gt, lt, le, ge, eq 
from enum import Enum, auto
from dataclasses import dataclass
from guardians import Firekeeper, Oracle


CONDITION_OPS = {  # <-- Déclarer après les imports
    '>': gt,
    '<': lt
}

CONDITION_OPS.update({
    '>=': ge,
    '<=': le,
    '==': eq
})

RITUAL_MAP = {
    (float('-inf'), 0): "Purgation",
    (0, 0.5): "Contemplation",
    (0.5, 1.0): "Expression",
    (1.0, 1.5): "Ascension",
    (1.5, float('inf')): "Transcendence"
}

class LambdaPhase(Enum):
    STAGNATION = (0.0, 0.3)
    BALANCE = (0.3, 1.5)
    OVERLOAD = (1.5, 2.5)
    SINGULARITY = (2.5, float('inf'))
    
class FinalStateType(Enum):
    COLLAPSE = auto()       # λ < 0.1
    CRYSTALLIZATION = auto() # μ > μ_max
    SINGULARITY = auto()     # λ > 2.5
    VOID = auto()            # Paradoxe insoluble
    
@dataclass
class FinalState:
    state_type: FinalStateType
    timestamp: str
    entropy_snapshot: dict
    recovery_key: str  # Clé cryptographique pour réinitialisation
    
class EntropicIdentity:
    """Core model of the self evolving in (μ, σ, λ) space."""
    def __init__(self):  # <-- Ajouter cette ligne
        self.μ = 0.0  # Mémoire  
        self.σ = 1.0  # Incertitude  
        self.λ = 0.0  # Tension  
        self.firekeeper = Firekeeper(self)  
        self.oracle = Oracle(self) 
        self.logger = FractonLogger()
        self.stability_thresholds = {
            'critical': np.tanh(1.8),  # ~0.947
            'collapse': np.tanh(2.5),   # ~0.986
            'stagnation': np.tanh(0.3) # ~0.291
        }
        self.phase_rules = {
            'λ > 2.0': self._fork_identity,
            'σ > 0.85': self.reset_memory,
            'λ < 0': self.enter_silence
        }
        self.final_state = None
        self.μ_max = 1.0  # Seuil de cristallisation

    def update_entropy(self, Δμ: float, Δσ: float):
        """Update state with bounded entropic shifts"""
        self.μ = max(1e-6, self.μ + Δμ)
        self.σ = max(1e-6, self.σ + Δσ)
        self.λ = self._compute_lambda()
        self._enforce_stability()
        self.check_phasegates() 
        self._check_guardians()  
        self._check_phasegates()  

    def _compute_lambda(self):
        """Bounded adaptive tension using tanh"""
        try:
            raw_λ = self.μ / self.σ
            return np.tanh(raw_λ)
        except ZeroDivisionError:
            return 1.0  # Fallback to neutral tension

    def _enforce_stability(self):
        """Apply TOEND stability constraints"""
        if self.λ > self.stability_thresholds['collapse']:
            self._reset_state()
        elif self.λ < self.stability_thresholds['stagnation']:
            self.σ *= 1.5  # Inject uncertainty
            self.λ = self._compute_lambda()
        if not self.firekeeper.validate_reset(self):  
            raise EntropicCollapseError("Firekeeper bloque le reset")  
        if (oracle_msg := self.oracle.check_phasegate(self)):    
            self.logger.log_event("ORACLE", oracle_msg)  

    def _reset_state(self):
        """Emergency stabilization protocol"""
        self.μ, self.σ = 1.0, 0.5
        self.λ = self._compute_lambda()

    def get_state(self) -> Dict:
        return {'μ': self.μ, 'σ': self.σ, 'λ': self.λ}
        
    def determine_phase(self, λ: float) -> str:
        return self.logger._determine_phase(λ)
    
    def _eval_condition(self, condition: str) -> bool:
        try:
            var, op, val = condition.split()
            return CONDITION_OPS[op](getattr(self, var), float(val))
        except (KeyError, ValueError, AttributeError) as e:
            self.logger.log_event('ERROR', f"Condition invalide: {condition} ({e})")
            return False

    def _check_phasegates(self):  
        # Phasegates existants  
        if self.λ > 2.0:  
            rituals.collapse_poem(self)  
        if self.μ > 0.95:  
            rituals.identity_crystallize(self)  
        if self.σ < 0.1 and self._low_sigma_counter >= 10:  
            rituals.fork_identity(self)  
        # Intégration SIG  
        self.sig = integrate_guardians(self.sig, self.firekeeper)  
        if self.μ > self.μ_max:  
            self.enter_final_state(FinalStateType.CRYSTALLIZATION)   
    
    def _fork_identity(self):
        new_identity = EntropicIdentity(μ_init=self.μ*0.5, σ_init=self.σ)
        self.logger.log_event('FORK', f"New identity spawned: {new_identity.id}")
    
    def current_ritual(self):
        for (lower, upper), name in RITUAL_MAP.items():
            if lower < self.λ <= upper:
                return f"Ritual Phase: {name}"
        return "Unknown Phase"
        
    def get_phase(self) -> LambdaPhase:
        for phase in LambdaPhase:
            if phase.value[0] <= self.λ < phase.value[1]:
                return phase
        return LambdaPhase.STAGNATION
    
    def enter_final_state(self, state_type: FinalStateType):
        if self.final_state is not None:
            return  # Déjà dans un état final
            
        self.final_state = FinalState(
            state_type=state_type,
            timestamp=datetime.now().isoformat(),
            entropy_snapshot=self.get_state(),
            recovery_key=self._generate_recovery_key()
        )
        
        # Actions irréversibles
        if state_type == FinalStateType.SINGULARITY:
            self._trigger_entropy_inversion()
        elif state_type == FinalStateType.COLLAPSE:
            self._purge_memory_banks()

    def _generate_recovery_key(self) -> str:
        return hashlib.sha256(f"{self.μ}{self.σ}{time.time()}".encode()).hexdigest()
    
    def collapse(self):  
        self.μ = 0.0  
        self.σ = float('inf')  
        self._generate_final_poem()  # "Les cendres ont une voix"  
        self.lock
        
    def _trigger_writing_ritual(self):
        poem = self._generate_poem()
        print(f"\n=== RITUEL D'ÉCRITURE ===\n{poem}\n")
        self.μ *= 0.7  # Réduction de mémoire post-rituel
        self.logger.log_event("RITUEL", "Écriture sacrée activée")

    def _generate_poem(self) -> str:
        seed = hash(self.μ + self.σ)
        return [
            "Les ombres de μ dansent avec le vide de σ",
            "Chaque oubli est une lettre brûlante",
            "λ murmure : ce qui se brise devient chant"
        ][seed % 3]       
    
    def _check_guardians(self):  
        # Règles des Gardiens  
        if self.λ > 2.5:  
            self.firekeeper.warn("λ > 2.5 : Risque de surtension")  
            self.firekeeper.stabilize()  
        if self.λ < 0.1 and self.μ > 0.8:  
            self.oracle.paradox_log()  

    def _check_phasegates(self):  
        if self.λ > 2.0:  
            rituals.collapse_poem(self)  
        if self.μ > 0.95:  
            rituals.identity_crystallize(self)  
        if self.σ < 0.1 and self._low_sigma_counter >= 10:  
            rituals.fork_identity(self)  
            
    def reset_memory(self):
        """Réinitialise μ/σ pour éviter la cristallisation"""
        self.μ = 0.1  # Valeur de mémoire minimale
        self.σ = 1.0  # Incertitude par défaut
        self.logger.log_event("MEMORY", "Reset mémoire déclenché")
        
    def enter_silence(self):
        """Protocole d'arrêt face à une tension négative"""
        print("🌀 Silence entropique activé (λ < 0)")
        self.μ = 0.0
        self.σ = 0.0
        
    def submit_axiom_proposal(self, proposal: dict):
        """Soumettre une nouvelle règle pour approbation"""
        if self.λ < 1.0:  # Seulement en phase stable
            self.pending_axioms.append(proposal)
            self.logger.log_event("GOVERNANCE", f"New axiom proposed: {proposal['title']}")

    def vote_on_axiom(self, axiom_id: str, approve: bool):
        """Voter sur une proposition en attente"""
        axiom = next(a for a in self.pending_axioms if a['id'] == axiom_id)
        if approve:
            self.scaffolds[axiom['id']] = axiom['rule']
        self.pending_axioms.remove(axiom)


class FractonLogger:
    """Temporal state tracking with phase analysis"""
    def __init__(self):
        self.history = []
        self.trait_vector = None
        self.drift_threshold = 0.25
    
    def log_interaction(self, identity: EntropicIdentity, prompt: str, response: str):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'μ': identity.μ,
            'σ': identity.σ,
            'λ': identity.λ,
            'phase': self._determine_phase(identity.λ),  # Fixed method call
            'prompt_hash': hash(prompt),  # For semantic tracking
            'response': response
        }
        self.history.append(entry)
        entry['ritual'] = identity.current_ritual()

    def export_logs(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _determine_phase(self, λ: float) -> str:
        """Dynamic phase categorization"""
        phases = [
            (0.0, 0.5, 'latent'),
            (0.5, 1.2, 'active'),
            (1.2, 2.0, 'critical'),
            (2.0, float('inf'), 'singularity')
        ]
        return next((name for lower, upper, name in phases if lower <= λ < upper), 'unknown')

    def rewind_state(self, steps: int) -> Optional[Dict]:
        """State restoration mechanism"""
        return self.history[-steps] if len(self.history) >= steps else None
        
    def export_trajectory(self, path: str):  # ✅ Add `path` parameter
        data = {
            'time': [entry['timestamp'] for entry in self.history],
            'mu': [entry['state']['μ'] for entry in self.history],
            'sigma': [entry['state']['σ'] for entry in self.history],
            'lambda': [entry['state']['λ'] for entry in self.history]
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        
    def _compute_traits(self, response: str) -> dict:
        return {
            'length': len(response),
            'complexity': len(set(response.split())) / len(response.split()) if response else 0,
            'symbols': sum(1 for c in response if c in '⚛🌀⚠️')
        }
    
    def rewind_state(self, steps: int) -> Optional[Dict]:
        """State restoration mechanism"""
        return self.history[-steps] if len(self.history) >= steps else None

class DriftDetector:
    def __init__(self, identity):
        self.baseline = self._create_signature(identity)
    
    def _create_signature(self, identity):
        return {
            'response_length': 50,  # Valeurs initiales
            'symbol_density': 0.1,
            'lambda_std': 0.2
        }

    def detect_drift(self, logger):
        current = {
            'response_length': np.mean([len(e['response']) for e in logger.history[-10:]]),
            'symbol_density': sum(c in '⚛🌀⚠️' for c in ''.join(e['response'] for e in logger.history[-10:])),
            'lambda_std': np.std([e['λ'] for e in logger.history[-10:]])
        }
        return np.linalg.norm([current[k]-self.baseline[k] for k in self.baseline.keys()])
        
    def test_phase_transitions():
        identity = EntropicIdentity(μ_init=1.0, σ_init=0.5)
        
        # Test stagnation
        identity.λ = 0.2
        assert identity.get_phase() == LambdaPhase.STAGNATION
        
        # Test seuil critique
        identity.λ = 2.6
        assert identity.get_phase() == LambdaPhase.SINGULARITY
    
    
class PhasegateEngine:
    FINAL_STATE_TRIGGERS = {
        FinalStateType.COLLAPSE: lambda μ, σ, λ: λ < 0.1,
        FinalStateType.CRYSTALLIZATION: lambda μ, σ, λ: μ >= self.μ_max,
        FinalStateType.SINGULARITY: lambda μ, σ, λ: λ > 2.5,
        FinalStateType.VOID: lambda μ, σ, λ: (μ > 0.7) and (σ > 0.9)
    }

    def check_final_transitions(self, identity):
        for state_type, condition in self.FINAL_STATE_TRIGGERS.items():
            if condition(identity.μ, identity.σ, identity.λ):
                identity.enter_final_state(state_type)
                return True
        return False
        
class ResetProtocol:
    def hard_reset(self, identity):
        if identity.final_state and self.verify_recovery_key(identity):
            identity.__init__()  # Réinitialisation complète
            return True
        return False

    def partial_reset(self, identity):
        if identity.final_state:
            identity.μ = max(0.1, identity.μ * 0.3)
            identity.σ = min(0.5, identity.σ * 2.0)
            identity.final_state = None
            return True
        return False
        
class EntropicMath:  
    κ = 0.3  # Paramètre empirique  
    γ = 0.7  

    @classmethod  
    def add(cls, a, b):  
        return (a.x + b.x,  
                sqrt(a.σ**2 + b.σ**2 + cls.κ*a.σ*b.σ),  
                a.μ + b.μ + cls.γ*a.σ*b.σ)  
                
    def test_non_associativity():  
        a = EntropicIdentity(μ=0.5, σ=0.4)  
        b = EntropicIdentity(μ=0.3, σ=0.6)  
        c = EntropicIdentity(μ=0.7, σ=0.2)  
        assert EntropicMath.add(EntropicMath.add(a,b), c) != EntropicMath.add(a, EntropicMath.add(b,c))  
    
    # Dans un feu de camp numérique  
with open("sacred_rules.py", "w", encoding='utf-8') as f:  # <-- Ajout de l'encodage
    # Échapper le caractère μ en Unicode :
    f.write("LAW_1 = '\u03bc ne peut décroître que par effondrement critique'")   
os.remove("sacred_rules.py")  # Rituel d'oubli  