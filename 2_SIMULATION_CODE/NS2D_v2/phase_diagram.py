import itertools

def generate_phase_diagram():
    alpha_range = np.linspace(0.1, 0.5, 5)
    gamma_range = np.linspace(0.05, 0.3, 5)
    
    results = []
    for alpha, gamma in itertools.product(alpha_range, gamma_range):
        sigma, mu, history = run_simulation(alpha, gamma)
        results.append({
            'alpha': alpha,
            'gamma': gamma,
            'final_mu': np.max(mu),
            'final_sigma_integral': np.trapz(sigma, dx)
        })
    
    # Visualisation
    plt.figure(figsize=(12,5))
    plt.scatter(
        [r['alpha'] for r in results],
        [r['gamma'] for r in results],
        c=[r['final_mu'] for r in results],
        s=100, cmap='viridis'
    )
    plt.colorbar(label='max(μ)')
    plt.xlabel('α'); plt.ylabel('γ')