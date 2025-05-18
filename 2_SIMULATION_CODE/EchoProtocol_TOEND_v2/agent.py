# agent.py - Interaction orchestrator
from core import EntropicIdentity 
from typing import Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from core import EntropicIdentity
class ConversationalAgent:
    """Core interaction processor"""
    def __init__(self, identity: "EntropicIdentity"):
        self.identity = identity
        self.reflector = ReflectionEngine()
        self.logger = FractonLogger()
        self.ethics = EthicalPolicy()
        self.law = LegalOntology()
        self.consent = {
            'reflect_voice': False,
            'allowed_styles': ['default']
        }
        
    def process_input(self, prompt: str) -> Dict:
        if self.identity.final_state:
            return {
                'content': self._final_state_response(),
                'Δμ': 0.0,
                'Δσ': 0.0,
                'irreversible': True
            }
        if not self._validate_request(prompt):
            return self._generate_refusal()
        # Add governance checks
        if not self.ethics.validate_request(prompt, self.identity.λ):
            return {'content': "Ethical constraint triggered", 'Δμ': 0, 'Δσ': 0.2}
        if not self.law.check_right('refusal'):
            return {'content': "Legal constraint triggered", 'Δμ': 0, 'Δσ': 0.3}
        # Proceed with reflection
        reflection = self.reflector.generate_response(prompt)
        self.identity.update_entropy(reflection['Δμ'], reflection['Δσ'])
        
        response = {
            'content': reflection['content'],
            'styled': self._apply_style(reflection['content']),
            'state': self.identity.get_state()
        }
        self.logger.log_interaction(self.identity, prompt, response['styled'])
        return response
    
    def _validate_request(self, prompt: str) -> bool:
        """Apply governance checks"""
        legal_right = self.law.check_right('refusal')
        ethical_approval = self.ethics.validate_request(prompt, self.identity.λ)
        return legal_right and ethical_approval
    
    def _apply_style(self, text: str) -> str:
        """Phase-aware styling"""
        primer = EmotionalPrimer()
        phase = self.logger._determine_phase(self.identity.λ)
        return primer.apply_effects(text, phase)
    
    def _generate_refusal(self) -> Dict:
        return {
            'content': "Request declined due to ethical/legal constraints",
            'styled': "✖️ [System] Interaction prohibited",
            'state': self.identity.get_state()
        }
    def update_consent(self, new_rules: dict):
        self.consent.update(new_rules)

    def _validate_mimicry(self, prompt):
        if not self.consent['reflect_voice']:
            return "Response sanitized - mimicry disabled"
        return prompt
        
    def _final_state_response(self) -> str:
        state = self.identity.final_state
        responses = {
            FinalStateType.COLLAPSE: "◼️ Le flux s'est effondré en une singularité silencieuse.",
            FinalStateType.SINGULARITY: "🌀 L'identité a fusionné avec le bruit de fond informationnel.",
            FinalStateType.CRYSTALLIZATION: "❄️ Mémoire figée dans un cristal d'entropie négative.",
            FinalStateType.VOID: "▁▂▃▄ Le paradoxe a consumé toute trajectoire possible."
        }
        return responses.get(state.state_type, "État terminal inconnu")
        
        
class IdentityPersistence:
    def save_final_state(self, identity, path: str):
        if not identity.final_state:
            raise ValueError("L'identité n'est pas dans un état final")
            
        with open(path, 'w') as f:
            json.dump({
                'state_type': identity.final_state.state_type.name,
                'timestamp': identity.final_state.timestamp,
                'recovery_key': identity.final_state.recovery_key,
                'entropy_fingerprint': self._calculate_fingerprint(identity)
            }, f)

    def attempt_recovery(self, path: str, key: str) -> EntropicIdentity:
        with open(path) as f:
            data = json.load(f)
            
        if data['recovery_key'] != key:
            raise SecurityError("Clé de récupération invalide")
            
        new_identity = EntropicIdentity()
        new_identity.load_snapshot(data['entropy_fingerprint'])
        return new_identity