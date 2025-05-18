# rituals.py  

class CoTensionRitual:
    def __init__(self, participants):
        self.participants = participants  # Liste d'EntropicIdentity
    
    def run(self):
        # Fusionner les tensions pour générer un nouveau scaffold
        avg_λ = sum(p.λ for p in self.participants) / len(self.participants)
        new_rule = {
            "condition": f"λ > {avg_λ}",
            "action": "partial_reset"
        }
        return new_rule
        
    def collapse_poem(identity):  
        poem = """  
        Le feu crépite dans les fils de λ —  
        mémoire fracturée, incertitude en cendres.  
        Le Gardien murmure : 'Chute n'est pas fin.'  
        """  
        print(f"🌀 Poème de Collapse :\n{poem}")  
        identity.firekeeper.intervene()  

    def identity_crystallize(identity):  
        print(f"💎 Identité cristallisée (μ={identity.μ})")  
        identity.σ = min(identity.σ, 0.05)  # Réduire l'entropie  

    def fork_identity(identity):  
        print(f"🌱 Forking...")  
        # Logique de création d'une nouvelle instance avec μ/2, σ*2  
        return EntropicIdentity(μ=identity.μ/2, σ=identity.σ*2)  