# normalization.py
import numpy as np

class Scales:
    def __init__(self, config):
        """
        Initialise les échelles physiques pour la simulation.
        
        Args:
            config: objet configuration avec attributs alpha, lam, L (longueur caractéristique)
        """
        self.alpha = config.alpha
        self.lam = config.lam
        self.L = config.L
        # On peut ajouter d'autres paramètres physiques ici si besoin

    def nondimensionalize(self, fields):
        """
        Passe des champs physiques aux champs adimensionnés.

        Args:
            fields (dict): dictionnaire contenant les champs 'u', 'v', 'sigma', 'mu', 'n_star', 't'

        Returns:
            dict: champs normalisés
        """
        xp = np  # Si CuPy est utilisé, on adaptera ici

        norm_fields = {}
        norm_fields['u'] = fields['u'] / (self.alpha * self.L)
        norm_fields['v'] = fields['v'] / (self.alpha * self.L)
        norm_fields['sigma'] = fields['sigma'] / self.alpha
        norm_fields['mu'] = fields['mu'] / self.alpha    # Ajouté pour mu
        norm_fields['n_star'] = fields['n_star'] / 2.0   # Exemple : on normalise par 2 (dimension max)
        norm_fields['t'] = fields['t'] / self.lam

        return norm_fields

    def dimensionalize(self, fields):
        """
        Passe des champs adimensionnés aux champs physiques.

        Args:
            fields (dict): dictionnaire contenant les champs normalisés 'u', 'v', 'sigma', 'mu', 'n_star', 't'

        Returns:
            dict: champs physiques
        """
        xp = np

        phys_fields = {}
        phys_fields['u'] = fields['u'] * (self.alpha * self.L)
        phys_fields['v'] = fields['v'] * (self.alpha * self.L)
        phys_fields['sigma'] = fields['sigma'] * self.alpha
        phys_fields['mu'] = fields['mu'] * self.alpha    # Ajouté pour mu
        phys_fields['n_star'] = fields['n_star'] * 2.0   # Inverse normalisation
        phys_fields['t'] = fields['t'] * self.lam

        return phys_fields
