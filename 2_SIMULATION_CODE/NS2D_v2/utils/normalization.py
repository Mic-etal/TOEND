"""
Physical scaling/normalization constants
"""
import numpy as np

class Scales:
    def __init__(self, config):
        # Velocity scale based on energy injection
        self.u_scale = np.sqrt(config.alpha / config.lmbda)
        
        # Time scale (large eddy turnover)
        self.t_scale = config.L / self.u_scale
        
        # Sigma scale (dimensionless)
        self.sigma_scale = 1.0
        
        # Strain rate scale
        self.s_scale = self.u_scale / config.L
        
    def nondimensionalize(self, fields):
        """Convert to dimensionless variables"""
        scaled = {}
        scaled['u'] = fields['u'] / self.u_scale
        scaled['v'] = fields['v'] / self.u_scale
        scaled['sigma'] = fields['sigma'] / self.sigma_scale
        scaled['t'] = fields['t'] / self.t_scale
        return scaled
        
    def dimensionalize(self, fields):
        """Convert back to physical units"""
        original = {}
        original['u'] = fields['u'] * self.u_scale
        original['v'] = fields['v'] * self.u_scale
        original['sigma'] = fields['sigma'] * self.sigma_scale
        original['t'] = fields['t'] * self.t_scale
        return original