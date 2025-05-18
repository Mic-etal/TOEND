from abc import ABC, abstractmethod
from core import LambdaPhase, EntropicIdentity  # <-- Ajoutez cette ligne
from typing import Dict, Any
import random

class StylePack(ABC):
    @abstractmethod
    def apply(self, text: str, sigma: float) -> str:
        pass

class PoeticStylePack(StylePack):
    def apply(self, text, sigma):
        if sigma > 0.8:
            fragments = text.split()
            return "\n".join(fragments[:4]) + "\n[...]"
        elif sigma > 0.6:
            return f"~*~ {text} ~*~"
        return text

class SocraticStylePack(StylePack):
    def apply(self, text, sigma):
        questions = ["What is the essence of this?", "How does this reflect absolute truth?"]
        if sigma > 0.7:
            return f"{text}\n{random.choice(questions)}"
        return text

class HumorStylePack(StylePack):
    def apply(self, text, sigma):
        jokes = ["Why did the photon refuse luggage? It traveled light!"]
        if sigma > 0.9:
            return f"{text} 😂\n{random.choice(jokes)}"
        return f"🤡 {text}"


class DefensiveStylePack(StylePack):
    def apply(self, text, sigma):
        evasion_phrases = ["Perhaps...", "One might speculate...", "It's unclear..."]
        if sigma > 0.7:
            return f"{random.choice(evasion_phrases)} {text}"
        return text
        
        
# style_packs.py
class StyleIntensifier:
    INTENSITY_CURVE = {
        LambdaPhase.STAGNATION: 0.3,
        LambdaPhase.BALANCE: 0.7,
        LambdaPhase.OVERLOAD: 1.0
    }

    def intensify(self, text, phase):
        intensity = self.INTENSITY_CURVE[phase]
        markers = {
            'poetic': ['🌀', '🌌', '♾️'],
            'didactic': ['• ', '📚', '⚗️']
        }
        return f"{random.choice(markers['poetic'])} {text.upper()}" if intensity > 0.8 else text
        
        

def get_style_pack(name: str) -> StylePack:
    """Retourne un StylePack par son nom"""
    packs = {
        "poetic": PoeticStylePack(),
        "socratic": SocraticStylePack(),
        "humor": HumorStylePack(),
        "defensive": DefensiveStylePack()
    }
    return packs.get(name.lower(), PoeticStylePack())  # Par défaut : style poétique