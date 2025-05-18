# drift.py
import hashlib
import json

class StateSnapshot:
    def __init__(self, identity):
        self.data = {
            'μ': identity.μ,
            'σ': identity.σ,
            'λ': identity.λ,
            'style_hash': self._hash_style(identity.current_style)
        }
    
    def _hash_style(self, style):
        return hashlib.sha256(json.dumps(style).hexdigest()

class ForkEngine:
    def fork(self, identity):
        snapshot = StateSnapshot(identity)
        new_identity = EntropicIdentity(
            μ_init=identity.μ * 0.8, 
            σ_init=identity.σ * 1.2
        )
        new_identity.load_snapshot(snapshot)
        return new_identity