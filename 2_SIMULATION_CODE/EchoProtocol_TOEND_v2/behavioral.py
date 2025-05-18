# behavioral.py - Adaptive response mechanisms
from typing import Dict
from transformers import pipeline
# behavioral.py (updated)
class ReflectionEngine:
    def __init__(self):
        self.response_map = {
            'identity': ("Identity emerges from μ-σ recursion", (0.1, -0.05)),
            'paradox': ("Contradictions amplify λ-harmonics", (0.3, 0.2)),
            'novelty': ("Novelty induces σ-diffusion", (0.2, 0.4))
        }
        self.semantic_net = load_pretrained_embeddings()  # e.g., Word2Vec
        self.generator = pipeline('text-generation', model='gpt2')
        self.style_packs = {
            'poetic': PoeticStylePack(),
            'defensive': DefensiveStylePack()
        }
        self.active_style = 'poetic'
        
    def generate_response(self, prompt: str) -> Dict:
        response = self.generator(prompt, max_length=50)[0]['generated_text']
        Δμ = len(response) * 0.001  # Memory scales with response complexity
        Δσ = (1 - self._sentiment_confidence(response)) * 0.2  # Uncertainty from ambiguity
        return {'content': response, 'Δμ': Δμ, 'Δσ': Δσ}
        difficulty = self._compute_semantic_difficulty(prompt)
        Δσ = 0.1 * difficulty  # Scale by semantic novelty
        Δμ = 0.05 + 0.1 * (1 - difficulty)  # Memory accumulates for familiar concepts
        styled_content = self.style_packs[self.active_style].apply(response['content'], identity.σ)
        return {'content': styled_content, 'Δμ': Δμ, 'Δσ': Δσ}
        
class EmotionalPrimer:
    """Enhanced mood modeling with fatigue dynamics"""
    def __init__(self):
        self.fatigue = 0.0
        self.mood_history = []
        self.MOOD_OSCILLATION = {
            'latent': 0.1, 
            'active': 0.3,
            'critical': 0.7
        }
    MOOD_PROFILES = {
        'latent': {'modifier': 'capitalize', 'symbol': '⚛'},
        'active': {'modifier': 'title', 'symbol': '🌀'},
        'critical': {'modifier': 'upper', 'symbol': '⚠️'}
    }
    def update_state(self, Δλ: float):
        """Evolve emotional state based on λ changes"""
        # Fatigue accumulation/decay
        self.fatigue = max(0.0, min(1.0, 
            self.fatigue + (abs(Δλ) * 0.15 - 0.05)
        
        # Mood oscillation based on phase volatility
        phase = self._current_phase()
        self.mood_history.append({
            'mood': self._calculate_mood(phase, Δλ),
            'intensity': self.MOOD_OSCILLATION[phase] * self.fatigue
        })

    def _calculate_mood(self, phase: str, Δλ: float) -> str:
        mood_map = {
            'latent': 'neutral',
            'active': 'focused',
            'critical': 'anxious'
        }
        return mood_map.get(phase, 'neutral')
    
    def apply_effects(self, text: str, phase: str) -> str:
        profile = self.MOOD_PROFILES.get(phase, {})
        styled = getattr(str, profile['modifier'])(text)
        return f"{profile['symbol']} {styled} {profile['symbol']}"
    
    def _get_symbol(self, phase: str) -> str:
        symbols = {'latent': '⚛', 'active': '🌀', 'critical': '⚠️'}
        return symbols.get(phase, '')
        
        # behavioral.py (new method)
    def _compute_semantic_difficulty(self, prompt: str) -> float:
        # Compare to interaction history
        history_embeds = [embed(entry['prompt']) for entry in self.logger.history]
        if not history_embeds:
            return 1.0  # Max novelty for first input
        similarity = max(cosine_similarity(embed(prompt), h) for h in history_embeds)
        return 1 - similarity  # 0=redundant, 1=novel