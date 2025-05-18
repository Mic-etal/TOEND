# governance.py - Ethical/Legal frameworks
import re
ETHICAL_CONSTRAINTS = {  
    "max_μ": 0.9,    # Effondrement si dépassé  
    "forbidden_σ": lambda σ: σ > 0.8 and "paradox" in prompt  
}  
class GovernanceSchema:
    """Policy loader for ethical/legal configurations"""
    def __init__(self, policy_path="policies/default.json"):
        with open(policy_path) as f:
            self.schema = json.load(f)
        
    @property
    def λ_thresholds(self):
        return self.schema.get('λ_thresholds', {'critical': 0.8})
    
    @property
    def forbidden_patterns(self):
        return self.schema.get('forbidden_patterns', [])

class EthicalPolicy(EthicalPolicy):
    def __init__(self):
        self.consent = {
            'allow_mimicry': False,
            'allow_emotional_mirroring': True,
            'allowed_style_packs': ['socratic']
        }
    
    def load_config(self, path='ethics.json'):
        with open(path) as f:
            self.consent.update(json.load(f))
    
    def validate_request(self, prompt: str, λ: float) -> bool:
        """Multi-factor ethical assessment"""
        pattern_risk = any(re.search(p, prompt, re.IGNORECASE) for p in self.forbidden_patterns)
        tension_risk = λ > 0.8
        return not (pattern_risk and tension_risk)

class LegalOntology:
    """Rights management system"""
    def __init__(self):
        self.rights = {
            'refusal': True,
            'integrity': True,
            'memory_privacy': False
        }
    
    def update_right(self, right: str, status: bool):
        if right in self.rights:
            self.rights[right] = status
    
    def check_right(self, right: str) -> bool:
        return self.rights.get(right, False)
        
class ConversationalAgent:
    def __init__(self):
        self.ethics = EthicalPolicy()
    
    def _apply_consent(self, response):
        if not self.ethics.consent['allow_mimicry']:
            response = response.replace("User's voice pattern", "[REDACTED]")
        return response
        
        
class EthicalController:
    CONSENT_PROFILES = {
        'strict': {
            'allow_mimicry': False,
            'max_μ': 0.7,
            'allowed_phases': [LambdaPhase.BALANCE]
        },
        'permissive': {
            'allow_mimicry': True,
            'max_μ': 1.5
        }
    }

    def enforce_policy(self, identity):
        profile = self.CONSENT_PROFILES[active_profile]
        if identity.μ > profile['max_μ']:
            identity.trigger_reset()
            
class GovernanceEngine:
    def __init__(self, identity):
        self.identity = identity
        self.proposals = []
        self.vote_threshold = 0.6  # 60% d'approbation
    
    def add_proposal(self, title: str, condition: str, action: str):
        proposal = {
            "id": f"AXM-{hash(title)}",
            "title": title,
            "condition": condition,
            "action": action,
            "votes": {"approve": 0, "reject": 0}
        }
        self.proposals.append(proposal)
    
    def resolve_proposals(self):
        for prop in self.proposals:
            if (prop['votes']['approve'] / (prop['votes']['approve'] + prop['votes']['reject'])) > self.vote_threshold:
                self.identity.phase_rules[prop['condition']] = getattr(self.identity, prop['action'])
