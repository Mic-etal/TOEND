# simulation.py - System testing
import random
import json
from agent import ConversationalAgent
agent = ConversationalAgent(EntropicIdentity())
class StressEvaluator:
    """Entropic stability tester"""
    TEST_PROFILES = {
        'paradox_storm': {
            'Δμ_range': (-0.5, 1.0),
            'Δσ_range': (-0.3, 1.5)
        },
        'memory_overload': {
            'Δμ_range': (0.7, 2.0),
            'Δσ_range': (-1.0, 0.2)
        }
    }
    def __init__(self):
        self.guardians = [Firekeeper(), Oracle()]
        
        
    def execute_test(self, identity: EntropicIdentity, test_type: str) -> Dict:
        """Run configured stress test"""
        profile = self.TEST_PROFILES[test_type]
        λ_history = []
        for _ in range(10):
            Δμ = random.uniform(*profile['Δμ_range'])
            Δσ = random.uniform(*profile['Δσ_range'])
            identity.update_entropy(Δμ, Δσ)
            λ_history.append(identity.λ)
        return {
            'λ_trajectory': λ_history,
            'phase_changes': [identity.determine_phase(λ) for λ in λ_history],
            'max_λ': max(λ_history),
            'min_λ': min(λ_history),
            'mean_λ': sum(λ_history)/len(λ_history)
        }
        for guardian in self.guardians:
            if msg := guardian.intervene(identity.σ, identity.λ):
                identity.update_entropy(Δμ=-0.2, Δσ=-0.3)
                results['interventions'].append(msg)
    
    def generate_chaos_scenario(self):
        """Randomly inject μ/σ shocks"""
        scenarios = [
            {'Δμ': random.uniform(-0.5, 1.5), 'Δσ': random.uniform(-0.3, 2.0)},
            {'Δμ': 2.0, 'Δσ': -0.8}  # Extreme memory saturation
        ]
        return random.choice(scenarios)
    
    def resilience_score(self, λ_history: List[float]) -> float:
        """Quantify system stability under stress"""
        recovery_time = len([λ for λ in λ_history if λ < 0.5])
        return 1 - (recovery_time / len(λ_history))
    
    def load_profile(self, json_path: str):
        with open(json_path) as f:
            self.TEST_PROFILES = json.load(f)
            
    def load_policy_config(self, json_path: str):
        with open(json_path) as f:
            config = json.load(f)
        self.forbidden_patterns = config.get('forbidden_patterns', self.forbidden_patterns)
        
    def run_simulation(prompt):
        identity = EntropicIdentity()
        detector = DriftDetector(identity)
        guardians = [Oracle(), Firekeeper()]
        
        response = agent.process_input(prompt)
        
        # Check for drift
        if (drift := detector.detect_drift(agent.logger)) > 0.5:
            agent.logger.log_event('DRIFT', f"Identity drift detected: {drift:.2f}")
        
        # Guardian interventions
        for guardian in guardians:
            if isinstance(guardian, Oracle):
                msg = guardian.check(identity.λ)
            elif isinstance(guardian, Firekeeper):
                msg = guardian.check(identity.μ, identity.σ)
            if msg:
                response['content'] = f"{msg}\n{response['content']}"
        
        identity._check_phasegates()
        return response
        

    def save_simulation(data, path='data/simulation_output.json'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

class CriticalityEngine:
    def run_kl_simulation(self, session_logs):
        past = [log for log in session_logs if log['t'] < 50]
        present = [log for log in session_logs if log['t'] >= 50]
        return self._compute_kl_divergence(past, present)

    def _compute_kl_divergence(self, p, q):
        # Implémentation de la divergence Kullback-Leibler
        ...
