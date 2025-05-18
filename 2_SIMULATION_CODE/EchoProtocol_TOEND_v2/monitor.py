# monitor.py
# -*- coding: utf-8 -*-
from time import sleep
import numpy as np

class StressEvaluator:
    """Classe pour exécuter des tests de stress sur l'identité entropique"""
    def execute_test(self, identity, test_name):
        # Implémentation basique pour passer l'erreur
        if test_name == "paradox_storm":
            identity.λ = 3.0  # Simulation de surtension
        elif test_name == "memory_overload":
            identity.μ = 1.2  # Dépassement de mémoire
        
        return {
            'max_λ': np.max([identity.λ, 2.5]),
            'phase_changes': ["STAGNATION", "OVERLOAD"]
        }
    def log_entropy_delta(identity):  
        return {  
            "μ": identity.μ, "σ": identity.σ, "λ": identity.λ,  
            "coherence_slope": (identity.λ - prev_λ) / time_delta  
        }  
    def check_phasegates(self, identity):
        if identity.λ > 2.5:
            print("⚠️ Phasegate Triggered: λ Collapse")
        if identity.μ > 1.0:
            print("⚠️ Memory Saturation")


class RealTimeDashboard:
    METRICS = ['μ', 'σ', 'λ', 'phase', 'drift_score']

    def display(self, identity, update_interval=1.0):
        while True:
            print(f"\nμ: {identity.μ:.2f} | σ: {identity.σ:.2f}")
            print(f"Phase: {identity.get_phase().name}")
            sleep(update_interval)
    # Dans RealTimeDashboard
    METRICS = ['μ', 'σ', 'λ', 'phase', 'pending_axioms', 'active_scaffolds']

    def display(self, identity, update_interval=1.0):
        # Ajouter cette ligne
        print(f"Scaffolds actifs: {len(identity.scaffolds)} | Propositions: {len(identity.pending_axioms)}")


class DriftDetector:
    def __init__(self):
        self.score = 0.0