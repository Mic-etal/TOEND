import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from tqdm import tqdm
from validation import SimulationValidator

class EntropicNS2DIntegrator:
    def __init__(self, config, physics, operators):
        self.config = config
        self.physics = physics
        self.ops = operators
        self.dt = config.dt_max
        self.iteration_count = 0

    def compute_adaptive_dt(self, u, v, n_star):
        """Calculate adaptive timestep based on:
        - CFL condition (velocity)
        - Diffusion stability (σ field)
        - Feedback timescale (μ evolution)
        """
        xp = cp if self.config.use_gpu else np
        
        # 1. CFL condition (velocity based)
        max_vel = max(xp.abs(u).max(), xp.abs(v).max())
        cfl_dt = 0.5 * self.config.dx / (max(max_vel) + 1e-12)
        
        # 2. Diffusion stability (σ field)
        sigma_diff_dt = 0.25 * (self.config.dx**2) / (self.config.eta + 1e-12)
        
        # 3. Feedback timescale (μ evolution)
        S = xp.clip(self.ops.strain_rate(u, v, n_star), 1e-6, 1e2)
        mu_feedback_dt = 1.0 / (self.config.gamma * S.mean() + 1e-12)
        
        # Take most restrictive condition
        dt = min(cfl_dt, sigma_diff_dt, mu_feedback_dt)
        
        # Apply configured bounds
        dt = xp.clip(dt, self.config.dt_min, self.config.dt_max)
        
        if xp.isnan(dt) or dt <= 0:
            print(f"Invalid dt={dt}, using dt_min={self.config.dt_min}")
            return self.config.dt_min
        
        return float(dt) if self.config.use_gpu else dt
        
    def save_crash_report(self, fields, diagnostics):
        """Save simulation state when crash occurs"""
        crash_data = {
            'fields': fields,
            'diagnostics': diagnostics,
            'last_dt': self.dt,
            'iteration': self.iteration_count
        }
        np.savez_compressed('crash_report.npz', **crash_data)

    def run(self, fields, diagnostics, max_steps=None):
        """Main simulation loop with safety checks"""
        validator = SimulationValidator(self.config)
        
        try:
            with tqdm(total=float(self.config.tmax), desc="Simulation") as pbar:
                current_time = float(fields.get('t', 0.0))
                tmax = float(self.config.tmax)
                
                while current_time < tmax:
                    fields = self.step(fields)
                    current_time = float(fields['t'])
                    
                    if not validator.validate_fields(fields):
                        print("\n⚠️ Invalid fields detected - stopping simulation")
                        break
                        
                    self.record_diagnostics(fields, diagnostics)
                    
                    try:
                        energy = diagnostics['energy'][-1] if diagnostics['energy'] else 0
                        energy_str = f"{energy:.3e}" if np.isfinite(energy) else "N/A"
                        pbar.set_postfix({
                            't': f"{current_time:.2f}",
                            'dt': f"{self.dt:.1e}", 
                            'E': energy_str
                        })
                        pbar.update(min(float(self.dt), tmax - current_time))
                    except Exception as e:
                        print(f"\nProgress bar update failed: {str(e)}")
                        continue
                    
                    if max_steps and len(diagnostics['time']) >= max_steps:
                        break
                        
        except Exception as e:
            print(f"\nSimulation crashed: {str(e)}")
            self.save_crash_report(fields, diagnostics)
        
        return fields, diagnostics

    def step(self, fields):
        """Full time integration step with stabilization for σ-μ dynamics"""
        xp = cp if self.config.use_gpu else np
        
        # 1. Calculate adaptive timestep
        self.dt = self.compute_adaptive_dt(fields['u'], fields['v'], fields.get('n_star', 1))
        
        # 2. Update core fields
        u, v = fields['u'], fields['v']
        sigma, mu = fields['sigma'], fields['mu']
        
        # Strain rate (clipped for stability)
        S = xp.clip(self.ops.strain_rate(u, v, fields.get('n_star', 1)), 1e-6, 1e2)
        
        # 3. σ dynamics (diffusion + nonlinear terms)
        sigma_diff = self.config.eta * self.ops.laplace(sigma)
        sigma_source = self.config.alpha * xp.sqrt(sigma) * S**1.5
        sigma_sink = -self.config.lam * sigma * S**(2-self.config.beta)
        sigma_new = sigma + self.dt * (sigma_diff + sigma_source + sigma_sink)
        
        # 4. μ dynamics (logistic growth with memory)
        mu_growth = self.config.gamma * sigma * (1 - mu/self.config.mu_max)
        mu_new = mu + self.dt * mu_growth
        
        # 5. Velocity update (with μ feedback)
        u, v = self.update_velocity(u, v, mu_new, S)
        
        # 6. Apply stabilization
        fields.update({
            'u': xp.nan_to_num(u),
            'v': xp.nan_to_num(v),
            'sigma': xp.clip(sigma_new, 1e-8, 1e2),
            'mu': xp.clip(mu_new, 0, self.config.mu_max),
            't': fields['t'] + self.dt
        })
        
        # 7. Optional spectral filtering
        if self.config.use_spectral_filter and (self.iteration_count % self.config.filter_interval == 0):
            fields['u'] = self.apply_spectral_filter(fields['u'])
            fields['v'] = self.apply_spectral_filter(fields['v'])
            fields['sigma'] = self.apply_spectral_filter(fields['sigma'])
        
        self.iteration_count += 1
        return fields

    def update_progress(self, pbar, fields, diagnostics):
        """Safe progress bar update"""
        try:
            energy = diagnostics['energy'][-1] if diagnostics['energy'] else 0
            energy_str = f"{energy:.3e}" if np.isfinite(energy) else "N/A"
            
            pbar.set_postfix({
                't': f"{fields['t']:.2f}",
                'dt': f"{self.dt:.1e}",
                'E': energy_str
            })
            pbar.update(min(self.dt, self.config.tmax - fields['t']))
        except:
            pbar.set_postfix({'status': 'updating...'})

    def handle_simulation_crash(self, fields, diagnostics, error_msg=None):
        """Graceful crash handling"""
        if error_msg:
            print(f"\nSimulation crashed: {error_msg}")
        
        # Sauvegarde d'urgence
        crash_data = {
            'fields': fields,
            'diagnostics': diagnostics,
            'last_valid_dt': self.dt,
            'error': error_msg
        }
        np.savez_compressed('crash_dump.npz', **crash_data)
        
        if self.config.debug_mode and hasattr(self.physics, 'debug_data'):
            np.save('physics_debug.npy', self.physics.debug_data)
            

    def apply_spectral_filter(self, field):
        """Mild low-pass filter for stability"""
        xp = cp if self.config.use_gpu else np
        fft = xp.fft.fft2 if len(field.shape) == 2 else xp.fft.fft
        
        field_hat = fft(field)
        if len(field.shape) == 2:
            kx = xp.fft.fftfreq(field.shape[0], d=self.config.dx)
            ky = xp.fft.fftfreq(field.shape[1], d=self.config.dx)
            ksq = kx[:,None]**2 + ky[None,:]**2
            filter_mask = xp.exp(-0.5*(ksq/self.config.k_cutoff**2))
        else:
            k = xp.fft.fftfreq(field.shape[0], d=self.config.dx)
            filter_mask = xp.exp(-0.5*(k/self.config.k_cutoff)**2)
        
        return (xp.fft.ifft(field_hat * filter_mask)).real
    def record_diagnostics(self, fields, diagnostics):
    # Existing metrics
    diagnostics['mean_sigma'].append(np.mean(fields['sigma']))
    diagnostics['mean_mu'].append(np.mean(fields['mu']))
    
    # TOEND-specific metrics
    diagnostics['sigma_mu_ratio'].append(np.mean(fields['sigma'] / (fields['mu'] + 1e-12)))
    diagnostics['x_variability'].append(np.std(fields['observable']))
    
    # Log-log scaling prep
    if self.iteration_count % 100 == 0:
        log_sigma = np.log10(fields['sigma'] + 1e-12)
        log_mu = np.log10(fields['mu'] + 1e-12)
        diagnostics['log_scaling'].append((log_sigma.mean(), log_mu.mean()))
    def update_velocity(self, u, v, mu, S):
    """Update velocity fields with μ feedback"""
    xp = cp if self.config.use_gpu else np
    
    # 1. Nonlinear terms
    u_conv = -0.5 * self.ops.div(u*u, u*v)
    v_conv = -0.5 * self.ops.div(u*v, v*v)
    
    # 2. Viscous terms
    u_visc = self.config.nu * self.ops.laplace(u)
    v_visc = self.config.nu * self.ops.laplace(v)
    
    # 3. μ feedback (gradient forcing)
    grad_mu_x, grad_mu_y = self.ops.grad(mu)
    Gamma = self.physics.memory_feedback(mu, S)
    
    # 4. Combined update
    u_new = u + self.dt * (u_conv + u_visc - Gamma * grad_mu_x)
    v_new = v + self.dt * (v_conv + v_visc - Gamma * grad_mu_y)
    
    return xp.nan_to_num(u_new), xp.nan_to_num(v_new)
